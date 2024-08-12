from test_water_potability import build_and_optimise_water_potability

write_only = True
time_limit = 1200
generate_water = True  # Set to True to generate water instances

for d_s in [0, 1]:
    for t_s in [0, 1]:
        # Generate the water instances
        water_data = []
        for n_w in [20, 25, 30]:
            # Generate the ensemble tree instances
            for n_e, d in [(6, 5), (7, 5), (8, 5)]:
                if generate_water:
                    scip = build_and_optimise_water_potability(
                        data_seed=d_s,
                        training_seed=t_s,
                        predictor_type="gbdt",
                        n_water_samples=n_w,
                        max_depth=d,
                        n_estimators_layers=n_e,
                        framework="torch",
                        build_only=True,
                    )
                    if write_only:
                        scip.writeProblem(
                            f"water_{n_w}_gbdt_{n_e}_{d}_sk_{d_s}_{t_s}.mps"
                        )
                    else:
                        scip.setParam("limits/time", time_limit)
                        scip.optimize()
                        water_data.append(
                            [
                                n_w,
                                "gbdt",
                                n_e,
                                d,
                                scip.getStatus(),
                                scip.getSolvingTime(),
                                scip.getNTotalNodes(),
                            ]
                        )
            # Generate the neural network instances
            for formulation in ["bigm", "sos"]:
                for n_e, d in [(7, 16), (8, 16)]:
                    if generate_water:
                        scip = build_and_optimise_water_potability(
                            data_seed=d_s,
                            training_seed=t_s,
                            predictor_type="mlp",
                            formulation=formulation,
                            n_water_samples=n_w,
                            layer_size=d,
                            n_estimators_layers=n_e,
                            framework="torch",
                            build_only=True,
                        )
                        if write_only:
                            scip.writeProblem(
                                f"water_{n_w}_mlp-{formulation}_{n_e}_{d}_torch_{d_s}_{t_s}.mps"
                            )
                        else:
                            scip.setParam("limits/time", time_limit)
                            scip.optimize()
                            water_data.append(
                                [
                                    n_w,
                                    "mlp",
                                    "torch",
                                    n_e,
                                    d,
                                    scip.getStatus(),
                                    scip.getSolvingTime(),
                                    scip.getNTotalNodes(),
                                ]
                            )

if not write_only:
    if generate_water:
        print(f"water: {water_data}", flush=True)
