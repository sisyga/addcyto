import numpy as np
from gillespy2 import Model, Species, Parameter, Reaction
from matplotlib import pyplot as plt


class AddCytotox_cont(Model):
    def __init__(self, parameter_values=None):
        # Initialize the model.
        Model.__init__(self, name="Additive cytotoxicity")

        # Define parameters.
        kappa1 = Parameter(name='kappa1', expression=.3)
        kappa0 = Parameter(name='alpha0', expression=.5)
        delta0 = Parameter(name='delta0', expression=0.01)
        deltac = Parameter(name='deltac', expression=.1)
        repairrate = Parameter(name='repairrate', expression=1)
        damagerate = Parameter(name='damagerate', expression=1)

        self.add_parameter([kappa0, kappa1, delta0, deltac, repairrate, damagerate])

        # Define molecular species.
        T = Species(name='Free_tumor_cell', initial_value=1)
        TI = Species(name='Immune_cell_attached', initial_value=0)
        D = Species(name='D', initial_value=0, mode='continuous')
        self.add_species([T, TI, D])

        # Define reactions.
        attach = Reaction(name="Immune_attachment", reactants={T: 1}, products={TI: 1}, rate=kappa1)
        detach = Reaction(name="Immune_detachment", reactants={TI: 1}, products={T: 1}, rate=kappa0)
        # propensity_function="alpha2/(1+pow(U,gamma))")
        damage = Reaction(name="Damage", reactants={TI: 1}, products={TI: 1, D: 1},
                          rate=damagerate)
        repair = Reaction(name="repair", reactants={D: 1}, products={},
                          rate=repairrate)
        death = Reaction(name='Cell death', reactants={T: 1}, products={}, propensity_function='delta0 * exp(D)')
        deathattached = Reaction(name='Cell death attached', reactants={TI: 1}, products={},
                                 propensity_function='delta0 * exp(D)')

        self.add_reaction([attach, detach, damage, repair, death, deathattached])
        self.timespan(np.linspace(0, 50, 101))


class AddCytotox(Model):
    def __init__(self, parameter_values=None):
        # Initialize the model.
        Model.__init__(self, name="Additive cytotoxicity")

        # Define parameters.
        deathratevalues = [0., 0., .5]
        imax = len(deathratevalues)
        deathrates = [Parameter(name='delta{}'.format(i), expression=d) for i, d in enumerate(deathratevalues)]
        kappa1 = Parameter(name='kappa1', expression=1)
        kappa0 = Parameter(name='kappa0', expression=1 / 0.3)
        repairrate = Parameter(name='repairrate', expression=6)
        damagerate = Parameter(name='damagerate', expression=6)
        proliferationrate = Parameter(name='proliferationrate', expression=1. / 24)

        self.add_parameter([kappa0, kappa1, repairrate, damagerate, proliferationrate] + deathrates)

        # Define molecular species.
        freetumorcells = [Species(name='Free 0 hit TC', initial_value=800)] + [
            Species(name='Free {} hit TC'.format(i + 1), initial_value=0) for i in range(imax - 1)]
        attachedctl = [Species(name='{} hit attached TC'.format(i), initial_value=0) for i in range(imax)]
        ctl = Species(name='Free CTL', initial_value=100)
        self.add_species(freetumorcells + attachedctl + [ctl])

        # Define reactions.
        attach = [
            Reaction(name="Immune_attachment{}".format(i), reactants={T: 1, ctl: 1}, products={TI: 1}, rate=kappa1)
            for i, (T, TI) in enumerate(zip(freetumorcells, attachedctl))]
        detach = [
            Reaction(name="Immune_detachment{}".format(i), reactants={TI: 1}, products={T: 1, ctl: 1}, rate=kappa0)
            for i, (T, TI) in enumerate(zip(freetumorcells, attachedctl))]
        # attach = Reaction(name="Immune_attachment", reactants={T: 1}, products={TI: 1}, rate=kappa1)
        # detach = Reaction(name="Immune_detachment", reactants={TI: 1}, products={T: 1}, rate=kappa0)
        # propensity_function="alpha2/(1+pow(U,gamma))")
        damage = [Reaction(name="Damage{}".format(i), reactants={TI1: 1}, products={TI2: 1}, rate=damagerate)
                  for i, (TI1, TI2) in enumerate(zip(attachedctl[:-1], attachedctl[1:]))]
        # damage = Reaction(name="Damage", reactants={TI: 1}, products={TI:1, D: 1},
        #               rate=damagerate)
        repairattached = [Reaction(name="Repairattached{}".format(i), reactants={TI1: 1}, products={TI2: 1},
                                   rate=repairrate) for i, (TI1, TI2) in
                          enumerate(zip(attachedctl[1:], attachedctl[:-1]))]
        repairfree = [Reaction(name="Repairfree{}".format(i), reactants={T1: 1}, products={T2: 1},
                               rate=repairrate) for i, (T1, T2) in
                      enumerate(zip(freetumorcells[1:], freetumorcells[:-1]))]
        death = [Reaction(name="Death{}".format(i), reactants={T: 1}, products={}, rate=deathrates[i])
                 for i, T in enumerate(freetumorcells)]
        deathattached = [
            Reaction(name="Deathattached{}".format(i), reactants={TI: 1}, products={ctl: 1}, rate=deathrates[i])
            for i, TI in enumerate(attachedctl)]
        proliferation = [
            Reaction(name='Proliferation', reactants={freetumorcells[0]: 1}, products={freetumorcells[0]: 2},
                     rate=proliferationrate)]

        self.add_reaction(
            attach + detach + damage + repairattached + repairfree + death + deathattached + proliferation)
        self.timespan(np.linspace(0, 24, 101))


if __name__ == '__main__':
    model = AddCytotox()
    results = model.run()
    results.plot()  # included_species_list=['Immune_cell_attached', 'D'])
    plt.show()
