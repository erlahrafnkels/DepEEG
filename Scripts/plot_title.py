from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
healthy = config.subject_classes.healthy


def make_plot_title(filename: str) -> str:
    name_split = filename[:-4].split("_")
    pretext_and_subject = name_split[0]
    sub_idx = name_split[0].find("S")
    subject = name_split[0][sub_idx:]
    pre_or_post = name_split[1]
    open_or_closed = name_split[2]
    h_or_d = ""

    if subject in healthy:
        h_or_d = "healthy"
    else:
        h_or_d = "depressed"
    if pre_or_post == "pre":
        pre_or_post = "Pretreatment"
    else:
        pre_or_post = "Posttreatment"
    if open_or_closed == "EO":
        open_or_closed = "eyes open"
    else:
        open_or_closed = "eyes closed"

    plot_title = (
        pretext_and_subject
        + " ("
        + h_or_d
        + "): "
        + pre_or_post
        + " recording with "
        + open_or_closed
    )

    return plot_title
