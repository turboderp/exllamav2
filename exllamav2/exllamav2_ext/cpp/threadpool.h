#ifndef _threadpool_h
#define _threadpool_h

#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include <chrono>

class ThreadPool
{

private:

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:

    ThreadPool(size_t threads)
    {
        stop = false;
        for (size_t i = 0; i < threads; ++i)
        {
            workers.emplace_back
            (
                [this]
                {
                    for (;;)
                    {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                            if (this->stop && this->tasks.empty()) return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                }
            );
        }
    }

    ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) worker.join();
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>
        (
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }
};

class Barrier
{

private:

    std::mutex mtx;
    std::condition_variable cv;
    int num_threads;
    int count;
    int generation;  // Track barrier cycles

public:

    Barrier(int num_threads) : num_threads(num_threads), count(0), generation(0) {}

    void arrive_and_wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        int current_generation = generation;

        if (++count == num_threads)
        {
            count = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, current_generation] { return current_generation != generation; });
        }
    }

    void reset(int new_num_threads)
    {
        std::unique_lock<std::mutex> lock(mtx);
        num_threads = new_num_threads;
        count = 0;
        generation++;  // Advance generation to unblock any waiting threads
        cv.notify_all();  // Wake up all waiting threads
    }
};

#endif