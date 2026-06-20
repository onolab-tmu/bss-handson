from pathlib import Path

from omegaconf import OmegaConf

base = OmegaConf.load("configs/bss.yaml")

for n_iter in [10, 50, 100]:
    config = OmegaConf.merge(
        base,
        OmegaConf.from_dotlist(
            [f"auxiva.n_iter={n_iter}", f"output_dir=results/bss_niter_{n_iter}"]
        ),
    )
    OmegaConf.save(config=config, f=Path(f"configs/bss_niter_{n_iter}.yaml"))

for reference_mic in [0, 1]:
    config = OmegaConf.merge(
        base,
        OmegaConf.from_dotlist(
            [
                f"auxiva.reference_mic={reference_mic}",
                f"output_dir=results/bss_refmic_{reference_mic}",
            ]
        ),
    )
    OmegaConf.save(config=config, f=Path(f"configs/bss_refmic_{reference_mic}.yaml"))

for model in ["laplace", "gauss"]:
    config = OmegaConf.merge(
        base,
        OmegaConf.from_dotlist(
            [f"auxiva.model={model}", f"output_dir=results/bss_model_{model}"]
        ),
    )
    OmegaConf.save(config=config, f=Path(f"configs/bss_model_{model}.yaml"))

for update_method in ["ip", "iss"]:
    config = OmegaConf.merge(
        base,
        OmegaConf.from_dotlist(
            [
                f"auxiva.update_method={update_method}",
                f"output_dir=results/bss_update_{update_method}",
            ]
        ),
    )
    OmegaConf.save(config=config, f=Path(f"configs/bss_update_{update_method}.yaml"))
