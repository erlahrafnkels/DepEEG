healthy = [
    "S2",
    "S3",
    "S9",
    "S10",
    "S12",
    "S13",
    "S14",
    "S15",
    "S19",
    "S20",
    "S24",
    "S25",
    "S30",
    "S32",
    "S38",
    "S39",
    "S42",
    "S46",
    "S29",
    "S6",
    "S23",
    "S47",
    "S49",
    "S53",
    "S55",
    "S56",
    "S57",
    "S60",
    "S61",
]


def make_plot_title(filename: str) -> str:
    name_split = filename[:-4].split("_")
    beginning = name_split[0]
    sub_idx = name_split[0].find("S")
    subject = name_split[0][sub_idx:]
    pre_or_post = name_split[1]
    open_or_closed = name_split[2]
    h_or_d = ""

    if subject in healthy:
        h_or_d = "Healthy"
    else:
        h_or_d = "Depressed"
    if pre_or_post == "pre":
        pre_or_post = ", pretreatment"
    else:
        pre_or_post = ", posttreatment"
    if open_or_closed == "EO":
        open_or_closed = ", eyes open"
    else:
        open_or_closed = ", eyes closed"

    plot_title = beginning + ": " + h_or_d + pre_or_post + open_or_closed

    return plot_title
