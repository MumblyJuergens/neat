#pragma once

#include <algorithm>
#include <ranges>
#include <neat/Brain.hpp>
#include <glube/AutoBuffer.hpp>
#include <glube/Attributes.hpp>
#include <glube/Program.hpp>
#include <glube/BitmapText.hpp>
#include <glm/vec2.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/color_space.hpp>
#undef GLM_ENABLE_EXPERIMENTAL
#include <mj/size.hpp>
#include <mj/algorithm.hpp>
#include <mj/nameof.hpp>

namespace neat::draw::gl
{

    class [[nodiscard]] Diagrammer final
    {
        struct Vertex
        {
            glm::vec2 position;
            glm::vec3 color;

            static glm::vec3 color_for(const NeuronType type)
            {
                switch (type)
                {
                case NeuronType::input: return { 1, 0, 0 };
                case NeuronType::output: return { 0, 1, 0 };
                case NeuronType::hidden: return { 0, 0, 1 };
                }
            }

            static glm::vec3 color_for(const Synapse &synapse)
            {
                if (synapse.enabled())
                {
                    return glm::rgbColor(glm::vec3{ 100.0f, 90.0f, 100.0f / synapse.weight() });
                }
                else
                {

                    return { 0.2f, 0.2f, 0.2f };
                }
            }
        };

        glube::AutoBuffer m_vertex_buffer;
        std::size_t m_neuron_vertices{};
        std::size_t m_synapse_vertices{};
        glube::Attributes m_vao;
        glube::Program m_program;
        glube::BitmapText m_text;

        glm::vec2 m_window_size{};

        public:

        [[nodiscard]] Diagrammer(const glm::ivec2 windowSize, const int textureUnit)
        {
            glEnable(GL_PROGRAM_POINT_SIZE);

            glube::Shader vertexShader{ glube::ShaderType::vertex,
            R"glsl(#version 460 core
            in vec2 position;
            in vec3 color;
            out vec3 vcolor;
            uniform mat4 projection;
            void main()
            {
                gl_Position = projection * vec4(position, 0.0, 1.0);
                gl_PointSize = 8;
                vcolor = color;
            })glsl" };
            glube::Shader fragmentShader{ glube::ShaderType::fragment,
            R"glsl(#version 460 core
            in vec3 vcolor;
            out vec4 ocolor;
            void main()
            {
                ocolor = vec4(vcolor, 1.0);
            })glsl" };
            m_program.add_shaders(vertexShader, fragmentShader);

            m_vao.add(m_program, m_vertex_buffer.buffer(), nameof(Vertex::position), &Vertex::position);
            m_vao.add(m_program, m_vertex_buffer.buffer(), nameof(Vertex::color), &Vertex::color);

            if (!m_text.load_bbf("sample/Roboto.bff"))
            {
                throw std::runtime_error{ "Font file could not be loaded" };
            }
            m_text.set_color({ 0.0f, 1.0f, 0.0f });

            configure(windowSize, textureUnit);
        }

        void configure(const glm::ivec2 windowSize, const int textureUnit)
        {
            m_window_size = windowSize;
            const auto projection = glm::ortho(0.0f, m_window_size.x, 0.0f, m_window_size.y, -1.0f, 1.0f);
            m_program.set_uniform("projection", projection);
            m_text.set_window_size(windowSize);
            m_text.set_texture_unit(textureUnit);
        }

        void build_diagram(const Brain &brain)
        {
            const auto layerCount = mj::sz_t(brain.layer_count());
            std::vector<int> sectionsInLayer(layerCount, 0);
            for (const auto &neuron : brain.neurons())
            {
                sectionsInLayer.at(mj::sz_t(neuron.layer())) += 1;
            }

            std::vector<Vertex> neuronVertices;
            std::vector<Vertex> synapseVertices;

            std::vector<int> nextSectionForLayer(layerCount, 0);
            const float layerWidth = m_window_size.x / static_cast<float>(layerCount);

            for (const auto &neuron : brain.neurons())
            {
                const auto layer = mj::sz_t(neuron.layer());
                const auto layerf = static_cast<float>(layer);
                const auto sectionf = static_cast<float>(nextSectionForLayer.at(layer)++);
                const float sectionHeight = m_window_size.y / static_cast<float>(sectionsInLayer.at(layer));
                Vertex vertex{
                    .position = {layerWidth * layerf + layerWidth / 2.0f, sectionHeight * sectionf + sectionHeight / 2.0f},
                    .color = Vertex::color_for(neuron.type()),
                };
                neuronVertices.push_back(vertex);
            }

            for (const auto &synapse : brain.synapses())
            {
                Vertex in = neuronVertices.at(mj::sz_t(synapse.in()));
                Vertex out = neuronVertices.at(mj::sz_t(synapse.out()));
                in.color = out.color = Vertex::color_for(synapse);
                synapseVertices.push_back(in);
                synapseVertices.push_back(out);
            }


            // m_text.prepare();
            // m_text.set_cursor(vertex.position + glm::vec2{ 0, 10 });
            // m_text.print(std::to_string(neuron.number()));

            // auto currentIndice = neuronCount;
            // for (const auto &synapse : synapses)
            // {
            //     indices.at(currentIndice++) = static_cast<unsigned int>(synapse.in());
            //     indices.at(currentIndice++) = static_cast<unsigned int>(synapse.out());

            //     const glm::vec2 pA = vertices.at(mj::sz_t(synapse.in())).position;
            //     const glm::vec2 pB = vertices.at(mj::sz_t(synapse.out())).position;
            //     auto distance = glm::max(pA, pB) - glm::min(pA, pB);
            //     auto location = glm::min(pA, pB) + (distance / 2.0f);

            //     m_text.set_cursor(location - glm::vec2{ 0, -10 });
            //     m_text.print(std::to_string(synapse.weight()));

            // }

            std::vector<Vertex> vertices;
            vertices.append_range(neuronVertices);
            vertices.append_range(synapseVertices);

            m_vertex_buffer.set(vertices.size() * sizeof(Vertex), vertices.data());

            m_neuron_vertices = neuronVertices.size();
            m_synapse_vertices = synapseVertices.size();
        }

        void draw_diagram()
        {
            m_program.activate();
            m_vao.activate();

            glDrawArrays(GL_LINES, m_neuron_vertices, m_synapse_vertices);
            glDrawArrays(GL_POINTS, 0, m_neuron_vertices);
        }
    };

} // End namespace neat::draw.