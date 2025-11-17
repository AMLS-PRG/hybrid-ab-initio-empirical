from deepmd_jax.train import train

# training an energy-force model
train(
      model_type='observables',              # Model type
      rcut=6.0,                              # Cutoff radius
      save_path='model.pkl',  # Path to save the trained model
      progress_path='progress.out',  # Path to save the training progress
      train_data_path=[
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_198K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_208K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_213K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_218K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_223K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_228K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_238K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_248K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_258K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_268K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_278K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_288K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_298K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_308K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_318K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_328K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_338K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_348K",
        "../../../../training-data/ab-initio/MB-pol/00_dlpoly_368K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_198K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_203K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_208K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_213K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_218K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_223K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_228K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_238K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_268K",
        "../../../../training-data/ab-initio/MB-pol/01_MB-polRefLiquidHighPressure_298K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_198K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_208K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_213K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_218K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_223K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_228K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_238K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_248K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_258K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_268K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_278K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_288K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_298K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_308K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_318K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_328K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_338K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_348K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_368K",
        "../../../../training-data/ab-initio/MB-pol/05_DPUnstable_500K",
        "../../../../training-data/ab-initio/MB-pol/06_DPLessAcc_288K",
        "../../../../training-data/ab-initio/MB-pol/06_DPLessAcc_298K",
        "../../../../training-data/ab-initio/MB-pol/06_DPLessAcc_308K",
        "../../../../training-data/ab-initio/MB-pol/06_DPLessAcc_318K",
        "../../../../training-data/ab-initio/MB-pol/06_DPLessAcc_328K",
        "../../../../training-data/ab-initio/MB-pol/06_DPLessAcc_338K",
      ], # Path (or a list of paths) to the training dataset
      train_data_path_obs=[
        '../../../../molecular-dynamics/ab-initio/MB-pol/260K/dataset',
      ],
      step=1000000,                          # Number of training steps
      print_loss_smoothing = 20,
      print_every = 1,
      temperature = 260,                   # Temperature in Kelvin
      target_observable = 1.0,            # density in g/cm^3
      batch_size_observable = 100,
      s_pref_obs = 0.02,
      l_pref_obs = 100,
      compress=True,
)
