#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <cmath>
#include <sched.h>
#include <pthread.h>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <syslog.h>

using namespace std::chrono;

std::mutex data_mutex;
std::mutex file_mutex;
std::atomic<bool> running(true);
std::atomic<bool> addition_enabled(true);  
int detected_fingers = 0;
bool stable = false;
int running_sum = 0;

void set_thread_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        perror("Error setting thread affinity");
    }
}

std::atomic<int> frame_count(0);
std::atomic<double> latest_fps(0.0);
duration<double, std::milli> gesture_wcet(0), addition_wcet(0), camera_wcet(0);
duration<double, std::milli> gesture_last_period(0), addition_last_period(0), logger_last_period(0);

std::ofstream metrics_file("metrics.csv", std::ios::out | std::ios::app);

template <typename Clock = high_resolution_clock>
auto time_diff(auto start, auto end) {
    return duration_cast<duration<double, std::milli>>(end - start);
}

void log_row(const std::string& metric, double value) {
    auto now = std::time(nullptr);
    std::lock_guard<std::mutex> lock(file_mutex);
    if (metrics_file.is_open()) {
        metrics_file << metric << "," << value << "," << std::ctime(&now);
        metrics_file.flush();
    }
}

void set_fifo_priority(int priority) {
    sched_param sch_params;
    sch_params.sched_priority = priority;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch_params) != 0) {
        perror("Failed to set FIFO priority. Run as root or with CAP_SYS_NICE");
    }
}

int count_fingers(std::vector<std::vector<cv::Point>>& contours) {
    if (contours.empty()) return 0;
    auto contour = *std::max_element(contours.begin(), contours.end(), [](auto& a, auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });

    std::vector<int> hull;
    cv::convexHull(contour, hull, false, false);
    if (hull.size() < 3) return 0;

    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hull, defects);

    int finger_count = 0;
    for (size_t i = 0; i < defects.size(); ++i) {
        auto& d = defects[i];
        cv::Point start = contour[d[0]];
        cv::Point end = contour[d[1]];
        cv::Point far = contour[d[2]];
        double a = cv::norm(end - start);
        double b = cv::norm(far - start);
        double c = cv::norm(end - far);
        double angle = acos((b * b + c * c - a * a) / (2 * b * c));
        if (angle <= CV_PI / 2 && d[3] > 10000) {
            finger_count++;
        }
    }

    if (finger_count == 0 && cv::contourArea(contour) > 2000) {
        return 1;
    }

    return finger_count + 1;
}

void* gesture_thread(void* arg) {
    set_thread_affinity(1);
    set_fifo_priority(80);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera failed to open." << std::endl;
        running = false;
        return nullptr;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

    int prev_fingers = -1;
    int stable_frame_count = 0;
    const int frames_to_stabilize = 15;
    int local_frame_counter = 0;

    cv::Mat frame, resized, hsv, mask;
    auto last_start = high_resolution_clock::now();
    auto loop_last_start = high_resolution_clock::now();
    auto fps_start_time = steady_clock::now();

    while (running) {
        auto loop_start = high_resolution_clock::now();
        gesture_last_period = time_diff(last_start, loop_start);
        last_start = loop_start;

        syslog(LOG_INFO, "Gesture thread period: %.3f ms", gesture_last_period.count());

        auto cap_start = high_resolution_clock::now();
        cap >> frame;
        auto cap_end = high_resolution_clock::now();

        auto cap_time = time_diff(cap_start, cap_end);
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            if (cap_time > camera_wcet) camera_wcet = cap_time;
        }

        if (frame.empty()) continue;

        frame_count++;
        local_frame_counter++;

        auto fps_now = steady_clock::now();
        if (duration_cast<seconds>(fps_now - fps_start_time).count() >= 2) {
            latest_fps = frame_count / 2.0;
            frame_count = 0;
            fps_start_time = fps_now;
        }

        if (local_frame_counter % 2 != 0) continue;

        cv::flip(frame, frame, 1);
        cv::resize(frame, resized, cv::Size(), 0.5, 0.5);

        if (hsv.empty()) hsv.create(resized.size(), CV_8UC3);
        if (mask.empty()) mask.create(resized.size(), CV_8UC1);

        cv::cvtColor(resized, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 50), mask);
        cv::dilate(mask, mask, {}, {-1, -1}, 2);
        cv::blur(mask, mask, {3, 3});

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        int fingers = 0;
        if (!contours.empty() && cv::contourArea(contours[0]) >= 2000) {
            fingers = count_fingers(contours);
        }

        if (fingers == prev_fingers && fingers != 0) {
            stable_frame_count++;
        } else {
            stable_frame_count = 0;
        }
        prev_fingers = fingers;

        if (stable_frame_count == frames_to_stabilize) {
            std::lock_guard<std::mutex> lock(data_mutex);
            detected_fingers = fingers;
            stable = true;
            stable_frame_count = 0;
        }

        std::string finger_text = "Fingers: " + std::to_string(fingers);
        std::string sum_text;
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            sum_text = "Sum: " + std::to_string(running_sum);
        }

        cv::putText(resized, finger_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(resized, sum_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

        cv::imshow("Hand Detection", resized);
        char key = (char)cv::waitKey(1);
        if (key == 'q') {
            running = false;
            break;
        } else if (key == 'p') {
            addition_enabled = !addition_enabled;
            std::cout << "[INFO] Addition thread " << (addition_enabled ? "enabled" : "paused") << std::endl;
        }

        auto loop_end = high_resolution_clock::now();
        auto exec_time = time_diff(loop_start, loop_end);
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            if (exec_time > gesture_wcet) gesture_wcet = exec_time;
        }
    }

    cap.release();
    return nullptr;
}

void* addition_thread(void* arg) {
    set_thread_affinity(1);
    set_fifo_priority(10);
    auto last_start = high_resolution_clock::now();

    while (running) {
        auto start = high_resolution_clock::now();
        addition_last_period = time_diff(last_start, start);
        last_start = start;

        syslog(LOG_INFO, "Addition thread period: %.3f ms", addition_last_period.count());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        {
            std::lock_guard<std::mutex> lock(data_mutex);
            if (stable && addition_enabled) {
                running_sum += detected_fingers;
                std::cout << "Added " << detected_fingers << " -> Total: " << running_sum << std::endl;
                log_row("Running_Sum", running_sum);
                stable = false;
            }

        }

        auto end = high_resolution_clock::now();
        auto exec_time = time_diff(start, end);
        if (exec_time > addition_wcet) addition_wcet = exec_time;
    }
    return nullptr;
}

void* logger_thread(void* arg) {
    set_fifo_priority(5);
    auto last_start = high_resolution_clock::now();

    while (running) {
        auto start = high_resolution_clock::now();
        logger_last_period = time_diff(last_start, start);
        last_start = start;

        syslog(LOG_INFO, "Logger thread period: %.3f ms", logger_last_period.count());

        std::this_thread::sleep_for(std::chrono::seconds(2));

        duration<double, std::milli> g_wcet, a_wcet;
        double fps;

        {
            std::lock_guard<std::mutex> lock(data_mutex);
            g_wcet = gesture_wcet;
            a_wcet = addition_wcet;
        }

        fps = latest_fps;

        log_row("Gesture_WCET(ms)", g_wcet.count());
        log_row("Addition_WCET(ms)", a_wcet.count());
        log_row("Camera_FPS", fps);
    }
    return nullptr;
}

int main() {
    openlog("GestureLogger", LOG_PID | LOG_CONS, LOG_USER);

    {
        std::lock_guard<std::mutex> lock(file_mutex);
        if (metrics_file.is_open())
            metrics_file << "Metric,Value,Timestamp\n";
    }

    pthread_t gesture_tid, addition_tid, logger_tid;

    if (pthread_create(&gesture_tid, nullptr, gesture_thread, nullptr) != 0) {
        perror("Failed to create gesture_thread");
        return 1;
    }

    if (pthread_create(&addition_tid, nullptr, addition_thread, nullptr) != 0) {
        perror("Failed to create addition_thread");
        return 1;
    }

    if (pthread_create(&logger_tid, nullptr, logger_thread, nullptr) != 0) {
        perror("Failed to create logger_thread");
        return 1;
    }

    pthread_join(gesture_tid, nullptr);
    pthread_join(addition_tid, nullptr);
    pthread_join(logger_tid, nullptr);

    cv::destroyAllWindows();

    if (metrics_file.is_open()) metrics_file.close();

    closelog();
    return 0;
}


// g++ syslog_rt.cpp -o sysrt `pkg-config --cflags --libs opencv4` -pthread -std=c++20
// sudo ./sysrt