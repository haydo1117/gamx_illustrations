
from gamx import Distribution, Link, GamFamily, CrSmooth, GamX, GamFormula
import gamx
import polars as pl
import os

df_iris = (
    pl.read_csv(
        os.path.join(
            'data',
            'iris.csv'
        )
    )
    .cast({
        'Species': pl.String,
    })
)

m2 = gamx.gam(
    df_iris,
    GamFormula(
        x = GamX(
            linear = ['Sepal.Width'],
            categorical = [('Species', 'Setosa')],
            smooth = [
                CrSmooth(
                    name = 'Sepal.Length',
                    knots = [2.0, 3.0, 4.0, 5.0, 6.0],
                    # sp = 1.5,
                    sp = None,
                ),
                CrSmooth(
                    name = 'Petal.Width',
                    knots = [0.3, 1.0, 1.5, 2.0, 2.2],
                    sp = None,
                ),
            ],
        ),
        y = 'Petal.Length',
        w = None,
        offset = None,
    ),
    GamFamily(
        distribution = Distribution.Gamma,
        link = Link.Log,
    ),
)

m2.p_vals

m2.plot('Sepal.Length')
m2.plot('Petal.Width')

m2.plot('Petal.Width', pars = gamx.PlotParameters(with_conf = False))


