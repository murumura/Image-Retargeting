#ifndef ARGS_H
#define ARGS_H
#include <memory>
#include <string_view>
#include <variant>
#include <vector>
#include <sstream>
#include <map>
#include <functional>
#include <utility>
#include <tuple>

template <typename Options>
class CommndLineParser: Options{
public:
    // each template parameter is of type  pointer to member of Opts
    using OptPtr = \
        std::variant<std::string Options::*, int Options::*, float Options::*, bool Options::*>;
    using Args = std::tuple<std::string, OptPtr, std::string>; 

    Options parse(int argc, const char* argv[])
    {
        std::vector<std::string_view> vargv(argv, argv+argc);
        for (int idx = 0; idx < argc; ++idx)
            for (auto& cbk : callbacks)
                cbk.second(idx, vargv);

        return static_cast<Options>(*this);
    }

    static std::unique_ptr<CommndLineParser> create(std::initializer_list<Args> args)
    {
        auto cmdOpts = std::unique_ptr<CommndLineParser>(new CommndLineParser());
        for (auto arg : args) cmdOpts->registerCallback(arg);
        return cmdOpts;
    }
    
    using callbackType = std::function<void(int, const std::vector<std::string_view>&)>;
    std::map<std::string, callbackType> callbacks;

    ~CommndLineParser() = default;
    CommndLineParser() = default;
    CommndLineParser(const CommndLineParser&) = delete;
    CommndLineParser(CommndLineParser&&) = delete;
    CommndLineParser& operator=(const CommndLineParser&) = delete;
    CommndLineParser& operator=(CommndLineParser&&) = delete;
private:
    auto registerCallback(std::string name, OptPtr optPtr, std::string desc)
    {
        callbacks[name] = [this, name, optPtr](int idx, const std::vector<std::string_view>& argv)
        {
            if (argv[idx] == name)
            {
                std::visit(
                    [this, idx, &argv](auto&& arg)
                    {
                        if (idx < argv.size() - 1)
                        {
                            std::stringstream value;
                            value << argv[idx+1];
                            value >> this->*arg;
                        }
                    },
                    optPtr);
            }
        };
    };

    auto registerCallback(Args p) { 
        return registerCallback(std::get<0>(p), std::get<1>(p), std::get<2>(p)); 
    }
};
#endif
