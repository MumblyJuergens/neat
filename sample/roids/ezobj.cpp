#include "ezobj.hpp"
#include <fstream>
#include <string>
#include <mj/string.hpp>


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
        auto split = mj::split(line, " ");

        if (split[0] == "v")
        {
            assert(split.size() == 4);
            vertices.push_back({ std::stof(split[1]), std::stof(split[2]) });
        }
        else if (split[0] == "f")
        {
            assert(split.size() == 4);
            result.push_back(vertices[std::stoul(split[1])]);
            result.push_back(vertices[std::stoul(split[2])]);
            result.push_back(vertices[std::stoul(split[3])]);
        }
        else if (split[0] == "l")
        {
            assert(split.size() == 3);
            result.push_back(vertices[std::stoul(split[1])]);
            result.push_back(vertices[std::stoul(split[2])]);
        }
    }
    return result;
}