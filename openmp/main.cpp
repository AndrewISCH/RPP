#include <omp.h>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>

std::uniform_real_distribution<double> global_dist(0.0, 1.0);

double generate_value(std::mt19937& gen) {
    return global_dist(gen);
}

double method_points(long long samples) {
    long long inside_count = 0;

    #pragma omp parallel
    {
        std::mt19937 gen(omp_get_thread_num() + time(NULL));
        long long local_count = 0;

        #pragma omp for
        for (long long i = 0; i < samples; ++i) {
            double x = generate_value(gen);
            double y = generate_value(gen);
            if (x * x + y * y <= 1.0) local_count++;
        }

        #pragma omp atomic
        inside_count += local_count;
    }

    return 4.0 * inside_count / samples;
}

double method_integral(long long samples) {
    double sum = 0.0;

    #pragma omp parallel
    {
        std::mt19937 gen(omp_get_thread_num() + time(NULL) + 100);
        double local_sum = 0.0;

        #pragma omp for
        for (long long i = 0; i < samples; ++i) {
            double x = generate_value(gen);
            local_sum += 1.0 / (1.0 + x * x);
        }

        #pragma omp atomic
        sum += local_sum;
    }

    return 4.0 * sum / samples;
}

int main(int argc, char* argv[]) {
    int num_threads = omp_get_max_threads();

    if (argc >= 2) {
        num_threads = std::atoi(argv[1]);
        if (num_threads < 1) {
            return 1;
        }
    }

    omp_set_num_threads(num_threads);

    long long total_samples = 1e8;

    double start1 = omp_get_wtime();
    double pi_points = method_points(total_samples);
    double end1 = omp_get_wtime();

    double start2 = omp_get_wtime();
    double pi_integral = method_integral(total_samples);
    double end2 = omp_get_wtime();

    std::cout << "Method 1 (circle):     PI = " << pi_points
              << " (Time: " << (end1 - start1) << " sec)" << std::endl;

    std::cout << "Method 2 (integral):   PI = " << pi_integral
              << " (Time: " << (end2 - start2) << " sec)" << std::endl;
    std::cout << "[OpenMP] Total time " << (end2 - start1) << " sec" << std::endl;

    return 0;
}
