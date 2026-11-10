"""Regenerate three existing figures with the inoculation-only label.

build_paper_assets.py is not rerun in full because its .tex output carries manual
fixes; this script re-executes only the three affected figure functions.
"""
import inspect

import build_paper_assets as B

D = B.load()
B.DISPLAY["nc_style"] = "Inoc. only"

src = inspect.getsource(B.fig_cliff).replace(
    'label="NC-style (F-indep.)"', 'label="Inoc. only (F-indep.)"')
ns = dict(B.__dict__)
exec(src, ns)
ns["fig_cliff"](D["e1"], None)

B.fig_rf(D["rfs"])
B.fig_gamma(D["gam"], D["rfs"])
print("regenerated fig2_e1_cliff, fig7_rf_asr, fig8_rf_gamma")
