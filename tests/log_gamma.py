
from gamx import Distribution, Link, GamFamily, CrSmooth, GamX, GamFormula
import gamx
import polars as pl
import os

df = (
    pl.read_parquet(
        os.path.join(
            'data',
            'log_gamma.parquet'
        )
    )
)

## m1
m1 = gamx.gam(
    df,
    GamFormula(
        x = GamX(
            linear = [],
            categorical = [],
            smooth = [
                CrSmooth(
                    name = 'x',
                    knots = [
                        1000000 * x
                        for x in [0.2, 0.4, 0.6, 0.8]
                    ],
                    sp = None,
                ),
            ],
        ),
        y = 'phi_low',
        w = None,
        offset = None,
    ),
    GamFamily(
        distribution = Distribution.Gamma,
        link = Link.Log,
    ),
)

m1.phi
m1.plot('x')


## m2
m2 = gamx.gam(
    df,
    GamFormula(
        x = GamX(
            linear = [],
            categorical = [],
            smooth = [
                CrSmooth(
                    name = 'x',
                    knots = [
                        1000000 * x
                        for x in [0.2, 0.4, 0.6, 0.8]
                    ],
                    sp = None,
                ),
            ],
        ),
        y = 'phi_high',
        w = None,
        offset = None,
    ),
    GamFamily(
        distribution = Distribution.Gamma,
        link = Link.Log,
    ),
)

m2.phi
m2.plot('x')

###
m1.plot('x')
m2.plot('x')
