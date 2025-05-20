#include <mpi.h>
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int total_particles = 500000;
    const int steps = 10000;
    int local_particles = total_particles / size;

    std::vector<Particle> particles(local_particles);

    unsigned int rand_state = rank + static_cast<unsigned int>(time(nullptr));

    double start = MPI_Wtime();

    for (int i = 0; i < local_particles; ++i) {
        for (int s = 0; s < steps; ++s) {
            unsigned int r = xorshift32(rand_state) & 3;
            particles[i].move(r);
        }
    }

    double local_sum_sq_dist = 0.0;
    for (const auto& p : particles) {
        local_sum_sq_dist += p.x * p.x + p.y * p.y;
    }

    double global_sum_sq_dist = 0.0;
    MPI_Reduce(&local_sum_sq_dist, &global_sum_sq_dist, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        double mean_sq_dist = global_sum_sq_dist / total_particles;
        std::cout << "[MPI] Mean square displacement after " << steps << " steps: " << mean_sq_dist << std::endl;
        std::cout << "Total time: " << (end - start) << " sec" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
