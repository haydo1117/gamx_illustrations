
from gamx import Distribution, Link, GamFamily, CrSmooth, GamX, GamxFormula, GamFormula
import gamx
import polars as pl
import os
import matplotlib.pyplot as plt

df_raw = (
    pl.read_parquet(
        os.path.join(
            'data',
            'log_gamma_unif_x.parquet'
        )
    )
    .sort('x')
)

## add excess
df = (
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

knots = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
]
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
                            for x in knots
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
print(phi1)

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
                        for x in knots
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

## plot m1 versus truth
m1p = m1.plot('x', display = False)

plt.plot(df['x'], df['mu'], label = 'truth', color='red', linestyle='-')
plt.plot(m1p.x, pl.Series(m1p.eta + m1.beta[0,0]).exp(), label = 'estimated', color='blue', linestyle='-')
plt.plot(m1p.x, pl.Series(m1p.eta_lower + m1.beta[0,0]).exp(), label = '95% confidence interval', color='grey', linestyle='--')
plt.plot(m1p.x, pl.Series(m1p.eta_upper + m1.beta[0,0]).exp(), color='grey', linestyle='--')

plt.xlabel('x')
plt.ylabel('mu')
plt.legend()
plt.show()

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
                            for x in knots
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
print(phi2)

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
                        for x in knots
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


## plot m2 versus truth
m2p = m2.plot('x', display = False)

plt.plot(df['x'], df['mu'], label = 'truth', color='red', linestyle='-')
plt.plot(m2p.x, pl.Series(m2p.eta + m2.beta[0,0]).exp(), label = 'estimated', color='blue', linestyle='-')
plt.plot(m2p.x, pl.Series(m2p.eta_lower + m2.beta[0,0]).exp(), label = '95% confidence interval', color='grey', linestyle='--')
plt.plot(m2p.x, pl.Series(m2p.eta_upper + m2.beta[0,0]).exp(), color='grey', linestyle='--')

plt.xlabel('x')
plt.ylabel('mu')
plt.legend()
plt.show()

## fit gam 
m1_gam = gamx.gam(
    df
        .filter(
            pl.col('phi_low_obs')
        )
        .with_columns(
            (pl.col('phi_low') - pl.col('excess')).alias('phi_low')
        ),
    GamFormula(
        x = GamX(
            linear = [],
            categorical = [],
            smooth = [
                CrSmooth(
                    name = 'x',
                    knots = [
                        1000000 * x
                        for x in knots
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

m1p_gam = m1_gam.plot('x', display = False)

plt.plot(df['x'], df['mu'], label = 'truth', color='red', linestyle='-')
plt.plot(m1p_gam.x, pl.Series(m1p_gam.eta + m1_gam.beta[0,0]).exp(), label = 'estimated', color='blue', linestyle='-')
plt.plot(m1p_gam.x, pl.Series(m1p_gam.eta_lower + m1_gam.beta[0,0]).exp(), label = '95% confidence interval', color='grey', linestyle='--')
plt.plot(m1p_gam.x, pl.Series(m1p_gam.eta_upper + m1_gam.beta[0,0]).exp(), color='grey', linestyle='--')

plt.xlabel('x')
plt.ylabel('mu')
plt.legend()
plt.show()

print(m1_gam.phi)

## high dispersion

m2_gam = gamx.gam(
    df
        .filter(
            pl.col('phi_high_obs')
        )
        .with_columns(
            (pl.col('phi_high') - pl.col('excess')).alias('phi_high')
        ),
    GamFormula(
        x = GamX(
            linear = [],
            categorical = [],
            smooth = [
                CrSmooth(
                    name = 'x',
                    knots = [
                        1000000 * x
                        for x in knots
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

m2p_gam = m2_gam.plot('x', display = False)

plt.plot(df['x'], df['mu'], label = 'truth', color='red', linestyle='-')
plt.plot(m2p_gam.x, pl.Series(m2p_gam.eta + m2_gam.beta[0,0]).exp(), label = 'estimated', color='blue', linestyle='-')
plt.plot(m2p_gam.x, pl.Series(m2p_gam.eta_lower + m2_gam.beta[0,0]).exp(), label = '95% confidence interval', color='grey', linestyle='--')
plt.plot(m2p_gam.x, pl.Series(m2p_gam.eta_upper + m2_gam.beta[0,0]).exp(), color='grey', linestyle='--')

plt.xlabel('x')
plt.ylabel('mu')
plt.legend()
plt.show()

print(m2_gam.phi)
