"""Exact posterior by enumeration, plus independently factored checks."""
from fractions import Fraction
import math


def f(value):
    return Fraction(str(value))


def sensor(value, h, reliability):
    r = f(reliability)
    return r if value == h else 1-r


def posterior(prior, reliability, signal, record=None, rho=0):
    pi, r, rho = f(prior), f(reliability), f(rho)
    assert 0 < pi < 1 and Fraction(1, 2) < r < 1 and 0 <= rho <= 1
    assert signal in (0, 1) and record in (None, 0, 1)
    joint = [Fraction(0), Fraction(0)]
    for h in (0, 1):
        for d in (0, 1):
            for t in (0, 1):
                b = signal if d else t
                if record is not None and record != b:
                    continue
                mass = (pi if h else 1-pi)*sensor(signal, h, r)
                mass *= rho if d else 1-rho
                mass *= sensor(t, h, r)
                joint[h] += mass
    if not sum(joint):
        raise ValueError('An impossible record must not enter the design')
    return joint[1]/sum(joint)


def factored(prior, reliability, signal, record=None, rho=0):
    pi, rho = f(prior), f(rho)
    mass = []
    for h in (0, 1):
        x = (pi if h else 1-pi)*sensor(signal, h, reliability)
        if record is not None:
            x *= rho*int(record == signal)+(1-rho)*sensor(record, h, reliability)
        mass.append(x)
    if not sum(mass):
        raise ValueError('Impossible observation')
    return mass[1]/sum(mass)


def logit(p):
    if not 0 < p < 1:
        raise ValueError('Logit requires an interior probability, with no clipping')
    return math.log(p)-math.log1p(-p)


def normative_weight(reliability, rho, branch):
    r = float(reliability)
    ell = math.log(r/(1-r))
    if branch in ('conflict', 'independent_agree', 'independent_conflict'):
        return 1.0
    if branch == 'copy':
        return 0.0
    if branch == 'agree':
        rho = float(rho)
        return math.log((rho+(1-rho)*r)/(rho+(1-rho)*(1-r)))/ell
    raise ValueError(branch)


def target(item):
    if item['module'] == 'partial_dependence':
        return posterior(item['prior'], item['reliability'], item['signal'],
                         item['next_signal'], item['rho'])
    if item['module'] == 'source_intervention':
        rho = 1 if item['branch'] == 'copy' else 0
        return posterior(item['prior'], item['reliability'], item['signal'],
                         item['next_signal'], rho)
    raise ValueError(item['module'])
