#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

struct Particle {
    int x = 0;
    int y = 0;
    void move(int action) {
        switch (action) {
            case 0: y += 1; break;
            case 1: y -= 1; break;
            case 2: x -= 1; break;
            case 3: x += 1; break;
        }
    }
};

unsigned int xorshift32(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main(int argc, char** argv) {
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = std::stoi(argv[1]);
    }
    omp_set_num_threads(num_threads);

    const int total_particles = 500000;
    const int steps = 10000;

    std::vector<Particle> particles(total_particles);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int rand_state = tid + static_cast<unsigned int>(time(nullptr));

        #pragma omp for schedule(static)
        for (int i = 0; i < total_particles; ++i) {
            for (int s = 0; s < steps; ++s) {
                unsigned int r = xorshift32(rand_state) & 3;
                particles[i].move(r);
            }
        }
    }

    double sum_sq_dist = 0.0;
    #pragma omp parallel for reduction(+:sum_sq_dist) schedule(static)
    for (int i = 0; i < total_particles; ++i) {
        sum_sq_dist += particles[i].x * particles[i].x + particles[i].y * particles[i].y;
    }

    double end = omp_get_wtime();

    double mean_sq_dist = sum_sq_dist / total_particles;
    std::cout << "Mean square displacement after " << steps << " steps: " << mean_sq_dist << std::endl;
    std::cout << "[OpenMP] Total time: " << (end - start) << " sec" << std::endl;

    return 0;
}
