from step2.constraints import run_ga_stage2

if __name__ == '__main__':
    best, fit, hist_best, hist_avg = run_ga_stage2(
        pop_size=80,
        generations=150,
        selection_method='tournament',
        crossover_method='arithmetic',
        mutation_rate=0.05,
        plot_convergence=True
    )