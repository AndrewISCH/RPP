#include <mpi.h>
#include <iostream>
#include <random>
#include <cmath>

std::uniform_real_distribution<double> global_dist(0.0, 1.0);

double generate_value(std::mt19937& gen) {
    return global_dist(gen);
}

double method_points(int samples, std::mt19937& gen) {
    long long int inside_count = 0;
    for (int i = 0; i < samples; ++i) {
        double x = generate_value(gen);
        double y = generate_value(gen);
        if (x * x + y * y <= 1.0) inside_count++;
    }
    return 4.0 * inside_count / samples;
}

double method_integral(int samples, std::mt19937& gen) {
    double sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        double x = generate_value(gen);
        sum += 1.0 / (1.0 + x * x);
    }
    return 4.0 * sum / samples;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long int total_samples = 1e8;
    long long int local_samples = total_samples / size;

    std::mt19937 gen(rank + time(NULL));

    // Замір часу
    double start_time = MPI_Wtime();

    double start1 = MPI_Wtime();
    double local_pi_points = method_points(local_samples, gen);
    double end1 = MPI_Wtime();

    double start2 = MPI_Wtime();
    double local_pi_integral = method_integral(local_samples, gen);
    double end2 = MPI_Wtime();

    double total_time = MPI_Wtime() - start_time;

    double global_pi_points = 0.0;
    double global_pi_integral = 0.0;

    MPI_Reduce(&local_pi_points, &global_pi_points, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_pi_integral, &global_pi_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Method 1 (circle):     PI = " << global_pi_points / size
                  << " (Time: " << (end1 - start1) << " sec)" << std::endl;

        std::cout << "Method 2 (integral):   PI = " << global_pi_integral / size
                  << " (Time: " << (end2 - start2) << " sec)" << std::endl;

        std::cout << "[MPI] Total time: " << total_time << " sec" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
