#ifndef CUSTOM_POOL_H
#define CUSTOM_POOL_H
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
namespace Image {

    class CustomThreadPool final {
    public:
        CustomThreadPool()
            : workerSize(std::thread::hardware_concurrency()),
              threads(new std::thread[std::thread::hardware_concurrency()])
        {
            createWorker();
        }

        explicit CustomThreadPool(std::size_t workerSize = std::thread::hardware_concurrency())
            : workerSize(workerSize ? workerSize : std::thread::hardware_concurrency()),
              threads(new std::thread[workerSize ? workerSize : std::thread::hardware_concurrency()])
        {
            createWorker();
        }

        ~CustomThreadPool()
        {
            waitForTasksFinished();
            isRunning = false;
            joinAllWorker();
        }

        // since std::thread objects are not copiable, it doesn't make sense for a
        // thread pool to be copiable.
        CustomThreadPool(const CustomThreadPool&) = delete;
        CustomThreadPool& operator=(const CustomThreadPool&) = delete;

        template <typename Fn>
        void pushTask(Fn&& f)
        {
            totalTasks++;
            {
                const std::scoped_lock lock(queueMutex);
                taskQueue.push(std::function<void()>(f));
            }
        }

        template <typename Fn, typename... Args>
        void pushTask(Fn&& f, Args&&... args)
        {
            pushTask([f, args...] { f(args...); });
        }

        template <typename Fn, typename... Args,
            typename = std::enable_if_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>, std::decay_t<Args>...>>>>
        std::future<bool> submit(Fn&& f, Args&&... args)
        {
            std::shared_ptr<std::promise<bool>> taskPromise(new std::promise<bool>);
            std::future<bool> future = taskPromise->get_future();
            pushTask([f, args..., taskPromise] {
                try {
                    f(args...);
                    taskPromise->set_value(true);
                }
                catch (...) {
                    try {
                        taskPromise->set_exception(std::current_exception());
                    }
                    catch (...) {
                    }
                }
            });
            return future;
        }

        template <typename Fn, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<Fn>, std::decay_t<Args>...>, typename = std::enable_if_t<!std::is_void_v<Ret>>>
        auto submit(Fn&& f, Args&&... args)
        {
            std::shared_ptr<std::promise<Ret>> taskPromise(new std::promise<Ret>);
            std::future<Ret> future = taskPromise->get_future();
            pushTask([f, args..., taskPromise] {
                try {
                    taskPromise->set_value(f(args...));
                }
                catch (...) {
                    try {
                        taskPromise->set_exception(std::current_exception());
                    }
                    catch (...) {
                    }
                }
            });
            return future;
        }

        template <typename T1, typename T2, typename F>
        void parallelForLoop(const T1& startIndex, const T2& indexAfterLast, const F& loop, std::size_t numBlocks = 0)
        {
            typedef std::common_type_t<T1, T2> T;
            T firstIndex = (T)startIndex;
            T lastIndex = (T)indexAfterLast;

            if (firstIndex == lastIndex)
                return;

            if (lastIndex < firstIndex) {
                T temp = lastIndex;
                lastIndex = firstIndex;
                firstIndex = temp;
            }

            lastIndex--;
            if (numBlocks == 0)
                numBlocks = workerSize;

            std::size_t totalSize = (std::size_t)(lastIndex - firstIndex + 1);
            std::size_t blockSize = (std::size_t)(totalSize / numBlocks);

            if (blockSize == 0) {
                blockSize = 1;
                numBlocks = (std::size_t)totalSize > 1 ? (std::size_t)totalSize : 1;
            }

            std::atomic<std::size_t> blocksRunning = 0;
            for (std::size_t t = 0; t < numBlocks; t++) {
                T start = ((T)(t * blockSize) + firstIndex);
                T end = (t == numBlocks - 1) ? lastIndex + 1 : ((T)((t + 1) * blockSize) + firstIndex);
                blocksRunning++;
                pushTask([start, end, &loop, &blocksRunning] {
                    loop(start, end);
                    blocksRunning--;
                });
            }

            while (blocksRunning != 0) {
                sleepOrYield();
            }
        }

        void waitForTasksFinished()
        {
            while (true) {
                if (!isPaused) {
                    if (totalTasks == 0)
                        break;
                }
                else {
                    if (getRunningTasksCount() == 0)
                        break;
                }
                sleepOrYield();
            }
        }

        std::size_t getTasksInQueue() const
        {
            const std::scoped_lock lock(queueMutex);
            return taskQueue.size();
        }

        std::size_t getRunningTasksCount() const
        {
            return totalTasks - getTasksInQueue();
        }

        std::size_t getTaskTotalSize() const
        {
            return totalTasks;
        }

        std::size_t getWorkerSize() const
        {
            return workerSize;
        }

    private:
        /* Create the threads in the pool and assign a worker to each thread. */
        void createWorker()
        {
            for (std::size_t i = 0; i < workerSize; i++) {
                threads[i] = std::thread(&CustomThreadPool::workerTask, this);
            }
        }

        void joinAllWorker()
        {
            for (std::size_t i = 0; i < workerSize; i++) {
                threads[i].join();
            }
        }

        bool popTask(std::function<void()>& task)
        {
            const std::scoped_lock lock(queueMutex);
            if (taskQueue.empty())
                return false;
            else {
                task = std::move(taskQueue.front());
                taskQueue.pop();
                return true;
            }
        }

        void workerTask()
        {
            while (isRunning) {
                std::function<void()> task;
                if (!isPaused && popTask(task)) {
                    task();
                    totalTasks--;
                }
                else {
                    sleepOrYield();
                }
            }
        }

        void sleepOrYield()
        {
            if (sleepDuration)
                std::this_thread::sleep_for(std::chrono::microseconds(sleepDuration));
            else
                std::this_thread::yield();
        }

        std::size_t sleepDuration = 1000;

        /* For synchronize access to the task queue by different threads.*/
        mutable std::mutex queueMutex = {};

        /* Indicating to the workers to keep running */
        std::atomic<bool> isRunning = true;

        /* Indicating whther stop worker from poping task from queue */
        std::atomic<bool> isPaused = false;

        /* Store tasks to be executed by the threads. */
        std::queue<std::function<void()>> taskQueue = {};

        /* The number of threads in the pool. */
        std::size_t workerSize;

        /* A smart pointer to manage the memory allocated for the threads. */
        std::unique_ptr<std::thread[]> threads;

        /* Keep track the total number of unfinished tasks(include task in queue / task been executing by thread) */
        std::atomic<std::size_t> totalTasks = 0;
    };
} // namespace Image
#endif