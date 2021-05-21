from math import cos, sqrt

from flask import Flask, request, render_template, flash
from markupsafe import Markup
from random import randint
import random
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='dev')


class Polynomial:
    """
    Contains polynomial coefficients
    """
    coefs: list  # coefs[i] is equal to coef before x^i

    def __init__(self, coefs):
        self.coefs = coefs

    def calculate(self, x):
        """
        :param x: x
        :return: returns p(x)
        """
        cur_pwr = 1.0
        result = 0.0
        for i, c in enumerate(self.coefs):
            result += c * cur_pwr
            cur_pwr *= x
        return result

    def to_html(self):
        res = ""
        for i, coef in enumerate(self.coefs):
            if res and coef > 0:
                res += "+"
            res += f"{coef: .3f}"
            if i:
                res += "x"
            if i > 1:
                res += f"<sup>{i}</sup>"
        return res


def random_polynomial(n):
    return Polynomial([random.uniform(- 0.5**i, 0.5 **i) for i in range(n)])


def approximated_function(x):
    return 2 * x * x - cos(x) - 1


def loss_function(polynomial):
    """
    :param polynomial: polynomial
    :return: calculates how good the approximation is (the smaller result corresponds to the better approximation)
    """
    res = 0.0
    for i in range(100):
        val = 2 / 100 * i
        appr = polynomial.calculate(val)
        res += (appr - approximated_function(val))**2
    return res


def mean_polynom(a, b):
    coefs = [(x + y) / 2 for x, y in zip(a.coefs, b.coefs)]
    return Polynomial(coefs)


def mutate(polynom: Polynomial):
    ind = randint(0, len(polynom.coefs) - 1)
    polynom.coefs[ind] += random.gauss(0, 0.5 ** ind)


def get_best_approximation(max_power: int):
    """
    :param max_power: maximum power of polynom
    :return: best polynomial approximation of x^2-cosx-1 on [0;2] found by genetic algorithm
    """
    n = max_power + 1
    population_size = 100
    population = [random_polynomial(n) for _ in range(population_size)]
    num_generations = 50
    best = population[0]
    for generation in range(num_generations):
        # generate new polynomials
        next_generation_size = population_size
        for _ in range(next_generation_size):
            first_polynom = population[randint(0, population_size - 1)]
            second_polynom = population[randint(0, population_size - 1)]
            population.append(mean_polynom(first_polynom, second_polynom))

        # mutation
        mutating_fraction = 0.1 * sqrt(n)
        for _ in range(int(mutating_fraction * len(population))):
            mutant_index = randint(0, len(population) - 1)
            mutate(population[mutant_index])

        # selection
        population.sort(key=lambda x: loss_function(x) * random.gauss(1, 0.1))
        population = population[:population_size]
        print(loss_function(min(population, key=loss_function)))
        best = min(*population, best, key=loss_function)
    return best


def draw_graph(pol: Polynomial, file: str):
    xs = [random.uniform(0, 2) for _ in range(100)]
    xs.sort()
    ys = [approximated_function(x) for x in xs]
    appr = [pol.calculate(x) for x in xs]
    plt.figure()
    plt.plot(xs, ys, label='x^2-cosx-1')
    plt.plot(xs, appr, label='approximation')
    plt.legend()
    plt.savefig(file)


@app.route('/', methods=('GET', 'POST'))
def index():
    messages = []
    pic = []
    if request.method == "POST":
        max_power = request.form['N']
        if not max_power:
            flash("You should enter N")
        else:
            polynom = get_best_approximation(int(max_power))
            messages = [Markup(f"Your result: {polynom.to_html()}")]
            draw_graph(polynom, 'static/result.png')
            pic = ['static/result.png']
    return render_template('index.html',
                           messages=messages, pic=pic)


if __name__ == '__main__':
    app.run()
