
from gamx import Distribution, Link, GamFamily, CrSmooth, GamX, GamxFormula
import gamx
import polars as pl
import os
import matplotlib.pyplot as plt

df_raw = (
    pl.read_parquet(
        os.path.join(
            'data',
            'log_gamma.parquet'
        )
    )
)

## add excess
df1 = (
    df_raw
        .with_columns(
            pl
            .when(
                pl.col('x') > 100000.0
            )
            .then(
                pl.lit(5000.0)
            )
            .when(
                pl.col('x') > 10000.0
            )
            .then(
                pl.lit(2000.0)
            )
            .otherwise(
                pl.lit(100.0)
            )
            .alias('excess')
        )
        .with_columns(
            (pl.col('phi_low') > pl.col('excess')).alias('phi_low_obs'),
            (pl.col('phi_high') > pl.col('excess')).alias('phi_high_obs'),
        )
)
df = (
    df1
        # .filter(
        #     pl.col('phi_low_obs').and_(pl.col('phi_high_obs')),
        #     pl.col('x') > 1000000.0 * 0.2,
        # )
)



## unknown dispersion
def search_phi(
    a: float,
    b: float,
    f: callable,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    phi = (a + b) / 2.0
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs.")
    iter = 0
    while (b - a) / 2.0 > tol:
        phi = (a + b) / 2.0
        fphi = f(phi)
        if fphi == 0:
            return phi
        elif fa * fphi < 0:
            b = phi
        else:
            a = phi
            fa = fphi
        iter += 1
        if iter >= max_iter:
            raise ValueError("Maximum number of iterations reached.")
    return phi

phi1 = search_phi(
    0.8,
    1.5,
    lambda phi: gamx.gamx(
        df
            .filter(
                pl.col('phi_low_obs')
            ),
        GamxFormula(
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
            a = 'excess',
            w = None,
            offset = None,
        ),
        GamFamily(
            distribution = Distribution.Gamma,
            link = Link.Log,
        ),
        phi = phi,
    ).chi_stat - 1.0
)

m1 = gamx.gamx(
    df
        .filter(
            pl.col('phi_low_obs')
        ),
    GamxFormula(
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
        a = 'excess',
        w = None,
        offset = None,
    ),
    GamFamily(
        distribution = Distribution.Gamma,
        link = Link.Log,
    ),
    phi = phi1,
)


phi2 = search_phi(
    1.0,
    2.0,
    lambda phi: gamx.gamx(
        df
            .filter(
                pl.col('phi_high_obs')
            ),
        GamxFormula(
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
            a = 'excess',
            w = None,
            offset = None,
        ),
        GamFamily(
            distribution = Distribution.Gamma,
            link = Link.Log,
        ),
        phi = phi,
    ).chi_stat - 1.0
)

m2 = gamx.gamx(
    df
        .filter(
            pl.col('phi_high_obs')
        ),
    GamxFormula(
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
        a = 'excess',
        w = None,
        offset = None,
    ),
    GamFamily(
        distribution = Distribution.Gamma,
        link = Link.Log,
    ),
    phi = phi2,
)


## plot
## unknown dispersion
plt.plot(df['x'], df['mu'], label = 'mu', color='blue', linestyle='-')
plt.plot(df.filter(pl.col('phi_low_obs'))['x'], m1.mu, label = 'phi_low', color='red', linestyle='-')
plt.plot(df.filter(pl.col('phi_high_obs'))['x'], m2.mu, label = 'phi_high', color='green', linestyle='-')
plt.legend()
plt.show()


## known dispersion

m1 = gamx.gamx(
    df
        .filter(
            pl.col('phi_low_obs')
        ),
    GamxFormula(
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
        a = 'excess',
        w = None,
        offset = None,
    ),
    GamFamily(
        distribution = Distribution.Gamma,
        link = Link.Log,
    ),
    phi = 1.2,
)

m2 = gamx.gamx(
    df
        .filter(
            pl.col('phi_high_obs')
        ),
    GamxFormula(
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
        a = 'excess',
        w = None,
        offset = None,
    ),
    GamFamily(
        distribution = Distribution.Gamma,
        link = Link.Log,
    ),
    phi = 2.0,
)

## plot
## known dispersion
plt.plot(df['x'], df['mu'], label = 'mu', color='blue', linestyle='-')
plt.plot(df.filter(pl.col('phi_low_obs'))['x'], m1.mu, label = 'phi_low', color='red', linestyle='-')
plt.plot(df.filter(pl.col('phi_high_obs'))['x'], m2.mu, label = 'phi_high', color='green', linestyle='-')
plt.legend()
plt.show()

##
df.filter(pl.col('phi_low_obs'))
df.filter(pl.col('phi_high_obs'))