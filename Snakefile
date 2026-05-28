configfile: "config/experiments.yaml"


def selection_names():
    seeds = config["selection_matrix"]["seeds"]
    metrics = config["selection_matrix"]["metrics"]
    window_size = config["selection_matrix"]["window_size"]
    names = [f"random_seed{seed}" for seed in seeds]
    for seed in seeds:
        for metric in metrics:
            names.append(f"{metric}_ws{window_size}_seed{seed}")
    return names


def split_ids():
    dataset_cfg = config["dataset"]
    if "split_ids" in dataset_cfg and dataset_cfg["split_ids"]:
        return [int(s) for s in dataset_cfg["split_ids"]]
    return [int(dataset_cfg["split_id"])]


def run_name(selection_name, split_id):
    molecule = config["molecule"]
    if selection_name.startswith("random_seed"):
        seed = selection_name.split("random_seed", 1)[1]
        return f"{molecule}_split{split_id:02d}_random_seed{seed}"
    metric_prefix = selection_name.rsplit("_ws", 1)[0]
    seed = selection_name.rsplit("_seed", 1)[1]
    return (
        f"{molecule}_split{split_id:02d}_{metric_prefix}"
        f"_ws{config['selection_matrix']['window_size']}_st{config['selection_matrix']['stride']}_seed{seed}"
    )


SEL_NAMES = selection_names()
SPLIT_IDS = split_ids()
RUN_NAMES = [run_name(sel, split_id) for split_id in SPLIT_IDS for sel in SEL_NAMES]
RUN_TO_META = {
    run_name(sel, split_id): {"selection_name": sel, "split_id": split_id}
    for split_id in SPLIT_IDS
    for sel in SEL_NAMES
}
DONE_DIR = config["outputs"]["snakemake_done_dir"]
AGG_JSON = config["outputs"]["aggregate_results_json"]
CKPT_ROOT = config["training"]["ckpt_root"]
PYTHON_BIN = config.get("execution", {}).get("python_bin", ".venv/bin/python")
RESUME = config.get("execution", {}).get("resume", True)


rule all:
    input:
        AGG_JSON


rule run_selection:
    output:
        result_json=f"{CKPT_ROOT}/experiment_metadata/{{run_name}}/result_summary.json",
        done=f"{DONE_DIR}/{{run_name}}.done",
    params:
        selection_name=lambda wc: RUN_TO_META[wc.run_name]["selection_name"],
        split_id=lambda wc: RUN_TO_META[wc.run_name]["split_id"],
        molecule=config["molecule"],
        data_path=config["dataset"]["data_path"],
        splits_dir=config["dataset"]["splits_dir"],
        ckpt_root=config["training"]["ckpt_root"],
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        n_train=config["training"]["n_train"],
        n_valid=config["training"]["n_valid"],
        energy_weight=config["training"]["energy_weight"],
        forces_weight=config["training"]["forces_weight"],
        save_every_epoch=config["training"]["save_every_epoch"],
        log_tb=config["training"]["log_tb"],
        print_freq=config["training"]["print_freq"],
        student_epochs=config["student"]["epochs"],
        student_learning_rate=config["student"]["learning_rate"],
        student_features=config["student"]["model"]["features"],
        student_max_degree=config["student"]["model"]["max_degree"],
        student_num_iterations=config["student"]["model"]["num_iterations"],
        student_num_basis_functions=config["student"]["model"]["num_basis_functions"],
        student_cutoff=config["student"]["model"]["cutoff"],
        student_charges=config["student"]["model"]["charges"],
        student_zbl=config["student"]["model"]["zbl"],
        window_size=config["selection_matrix"]["window_size"],
        stride=config["selection_matrix"]["stride"],
        train_fraction=config["selection_matrix"]["train_fraction"],
        seeds=" ".join(str(x) for x in config["selection_matrix"]["seeds"]),
        metrics=" ".join(config["selection_matrix"]["metrics"]),
        convert_to_ev=config["dataset"]["convert_to_ev"],
        python_bin=PYTHON_BIN,
        resume=RESUME,
    wildcard_constraints:
        run_name="|".join(RUN_NAMES),
    shell:
        (
            "{params.python_bin} -m src.snakemake_runner run-selection "
            "--selection-name {params.selection_name} "
            "--output-json {output.result_json} "
            "--done-file {output.done} "
            "--molecule {params.molecule} "
            "--data-path {params.data_path} "
            "--splits-dir '{params.splits_dir}' "
            "--split-id {params.split_id} "
            "--ckpt-root {params.ckpt_root} "
            "--num-epochs {params.num_epochs} "
            "--batch-size {params.batch_size} "
            "--learning-rate {params.learning_rate} "
            "--n-train {params.n_train} "
            "--n-valid {params.n_valid} "
            "--energy-weight {params.energy_weight} "
            "--forces-weight {params.forces_weight} "
            "--save-every-epoch {params.save_every_epoch} "
            "--log-tb {params.log_tb} "
            "--print-freq {params.print_freq} "
            "--student-epochs {params.student_epochs} "
            "--student-learning-rate {params.student_learning_rate} "
            "--student-features {params.student_features} "
            "--student-max-degree {params.student_max_degree} "
            "--student-num-iterations {params.student_num_iterations} "
            "--student-num-basis-functions {params.student_num_basis_functions} "
            "--student-cutoff {params.student_cutoff} "
            "--student-charges {params.student_charges} "
            "--student-zbl {params.student_zbl} "
            "--convert-to-ev {params.convert_to_ev} "
            "--window-size {params.window_size} "
            "--stride {params.stride} "
            "--train-fraction {params.train_fraction} "
            "--seeds {params.seeds} "
            "--metrics {params.metrics} "
            "--resume {params.resume}"
        )


rule aggregate:
    input:
        expand(f"{CKPT_ROOT}/experiment_metadata/{{run_name}}/result_summary.json", run_name=RUN_NAMES)
    output:
        AGG_JSON
    params:
        python_bin=PYTHON_BIN
    shell:
        "{params.python_bin} -m src.snakemake_runner aggregate --input-json {input} --output-json {output}"
