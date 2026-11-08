BG = "#DCE7E1"          # soft sage-green page background 
ACCENT = "#D6E6EF"      # soft light-blue surfaces for cards and controls
INK = "#20353C"         # deep blue-grey 
INK_ON_ACCENT = "#20353C"   # dark text on the light sage surface


def css_vars() -> str:
    return (":root{"
            f"--ust-bg:{BG};--ust-accent:{ACCENT};--ust-ink:{INK};--ust-on-accent:{INK_ON_ACCENT};"
            "}")


def _make_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ust_two_colour", [BG, ACCENT, INK])


CMAP = _make_cmap()
