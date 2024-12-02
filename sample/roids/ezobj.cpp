#include "ezobj.hpp"
#include <fstream>
#include <ranges>
#include <string>


// Yes, I am aware this is fragile dogshit, it took one minute to write for
// a sample, don't judge me.
std::vector<glm::vec2> load2D(std::string_view fileName)
{
    std::ifstream in{ fileName.data() };
    std::string line;
    std::vector<glm::vec2> vertices, result;
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        auto split = line | std::views::split(' ') | std::views::transform([](auto s) { return std::string_view{ s.data(),s.size() };});

        if (split[0] == 'v')
        {
        }
        else if (line[0] == 'f')
        {
        }
    }
    return result;
}