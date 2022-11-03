import gillespy2
import numpy as np
from gillespy2 import Model, Species, Parameter, Reaction
from gillespy2.solvers.numpy.ode_solver import ODESolver
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

class AddCytoChemo_simplified(Model):
    def __init__(self, parameter_values=None):
        # Initialize the model.
        Model.__init__(self, name="Additive cytotoxicity")

        # Define parameters.
        deathratevalues = [0., 0.1, .5]
        imax = len(deathratevalues)
        deathrates = [Parameter(name='delta{}'.format(i), expression=d) for i, d in enumerate(deathratevalues)]
        repairrate = Parameter(name='repairrate', expression=6)
        ctldamagerate = Parameter(name='damagerate', expression=.01)
        proliferationrate = Parameter(name='proliferationrate', expression=1. / 24)
        ctlchemokillrate = Parameter(name='ctlchemokillrate', expression=1)
        chemodamagerate = Parameter(name='chemodamagerate', expression=1)
        ctlrecruitmentrate = Parameter(name='CTLrecruitmentrate', expression=1)
        ctlremovalrate = Parameter(name='CTLremovalrate', expression=.01)

        self.add_parameter([repairrate, ctldamagerate, ctlchemokillrate, chemodamagerate,
                            proliferationrate, ctlremovalrate, ctlrecruitmentrate] + deathrates)

        # Define molecular species.
        tumorcells = [Species(name='0 hit TC', initial_value=1000)] + [
            Species(name='{} hit TC'.format(i + 1), initial_value=0) for i in range(imax - 1)]
        ctl = Species(name='CTL', initial_value=100)
        chemo = Species(name='Chemo', initial_value=1)
        self.add_species(tumorcells + [ctl] + [chemo])

        # Define reactions.
        ctldamage = [Reaction(name="Damage{}".format(i), reactants={TI1: 1, ctl: 1}, products={TI2: 1, ctl:1},
                              rate=ctldamagerate)
                  for i, (TI1, TI2) in enumerate(zip(tumorcells[:-1], tumorcells[1:]))]
        chemodamage = [Reaction(name="ChemoDamage{}".format(i), reactants={TI1: 1, chemo: 1}, products={TI2: 1, chemo:1},
                                rate=chemodamagerate)
                  for i, (TI1, TI2) in enumerate(zip(tumorcells[:-1], tumorcells[1:]))]
        repairfree = [Reaction(name="Repairfree{}".format(i), reactants={T1: 1}, products={T2: 1},
                               rate=repairrate) for i, (T1, T2) in
                      enumerate(zip(tumorcells[1:], tumorcells[:-1]))]
        death = [Reaction(name="Death{}".format(i), reactants={T: 1}, products={}, rate=deathrates[i])
                 for i, T in enumerate(tumorcells)]
        ctldeath = Reaction(name='CTLDeath', reactants={ctl: 1, chemo: 1}, products={chemo:1}, rate=ctlchemokillrate)
        ctlrecruitment = Reaction(name='CTLrecruitment', reactants={}, products={ctl: 1}, rate=ctlrecruitmentrate)
        ctlremoval = Reaction(name='CTLremoval', reactants={ctl:1}, products={}, rate=ctlremovalrate)
        proliferation = [
            Reaction(name='Proliferation', reactants={tumorcells[0]: 1}, products={tumorcells[0]: 2},
                     rate=proliferationrate)]

        self.add_reaction(
            chemodamage + ctldamage + repairfree + death + proliferation + [ctldeath, ctlrecruitment, ctlremoval])
        self.timespan(np.linspace(0, 100, 101))

class AddCytotox(Model):
    def __init__(self, parameter_values=None):
        # Initialize the model.
        Model.__init__(self, name="Additive cytotoxicity")

        # Define parameters.
        deathratevalues = [0., 0.1, .5]
        imax = len(deathratevalues)
        deathrates = [Parameter(name='delta{}'.format(i), expression=d) for i, d in enumerate(deathratevalues)]
        kappa1 = Parameter(name='kappa1', expression=1)
        kappa0 = Parameter(name='kappa0', expression=1 / 0.3)
        repairrate = Parameter(name='repairrate', expression=6)
        damagerate = Parameter(name='damagerate', expression=6)
        proliferationrate = Parameter(name='proliferationrate', expression=1. / 24)
        ctlchemokillrate = Parameter(name='ctlchemokillrate', expression=0.1)
        chemodamagerate = Parameter(name='chemodamagerate', expression=.02)


        self.add_parameter([kappa0, kappa1, repairrate, damagerate, proliferationrate] + deathrates)

        # Define molecular species.
        freetumorcells = [Species(name='Free 0 hit TC', initial_value=800)] + [
            Species(name='Free {} hit TC'.format(i + 1), initial_value=0) for i in range(imax - 1)]
        attachedctl = [Species(name='{} hit attached TC'.format(i), initial_value=0) for i in range(imax)]
        ctl = Species(name='Free CTL', initial_value=100)
        chemo = Species(name='Chemo', initial_value=100)
        self.add_species(freetumorcells + attachedctl + [ctl] + [chemo])

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


class CellDamages(Model):
    def __init__(self, parameter_values=None):
        # Initialize the model.
        Model.__init__(self, name="Cellular damages")

        # Define parameters.
        focirepairrate = Parameter(name='focirepairrate', expression=.5)
        hitrate = Parameter(name='hitrate', expression=.5)
        membranerepairrate = Parameter(name='membranerepairrate', expression=100)
        focidamagerate = Parameter(name='focidamagerate', expression=6)

        self.add_parameter([focirepairrate, hitrate, membranerepairrate, focidamagerate])

        # Define molecular species.
        ca2 = Species(name='Ca2', initial_value=0, mode='discrete')
        foci = Species(name='foci', initial_value=0, mode='discrete')
        self.add_species([ca2, foci])

        # Define reactions.
        # membrane rupture
        rupture = Reaction(name='membrane_rupture', reactants={}, products={ca2: 50}, rate=hitrate)
        membranerepair = Reaction(name='membrane_repair', reactants={ca2: 1}, products={}, rate=membranerepairrate)
        focidamage = Reaction(name='foci_damage', reactants={ca2: 1}, products={ca2: 1, foci: 1}, rate=focidamagerate)
        focirepair = Reaction(name='foci_repair', reactants={foci: 1}, products={}, rate=focirepairrate)
        # propensity_function="alpha2/(1+pow(U,gamma))")

        self.add_reaction([rupture, membranerepair, focidamage, focirepair])
        self.timespan(np.linspace(0, 50, 1001))


if __name__ == '__main__':
    fig_width_pt = 455.24411 / 3  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    params = {'axes.labelsize': 8,
              'font.sans-serif': 'Arial',
              'font.size': 8,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': False,
              'figure.figsize': fig_size}
    plt.rcParams.update(params)
    model = CellDamages()
    results = model.run(algorithm='SSA')
    # print(results)
    # results.plot(figsize=fig_size)  # included_species_list=['Immune_cell_attached', 'D'])
    # plt.yscale('log')
    plt.figure(figsize=(1.5, 1.25))
    plt.plot(results.data[0]['time'], results.data[0]['Ca2'], label=r'$c_{{\rm Ca}^{2+}}$', lw=1)
    plt.plot(results.data[0]['time'], results.data[0]['foci'], label=r'$n_{\rm foci}$', lw=1)
    plt.legend(borderpad=0.1, fontsize=8, frameon=True, handlelength=1, edgecolor='1.')
    plt.xlabel('Time (h)')
    plt.ylabel('')
    plt.ylim(-1, 20)
    plt.xlim(0, 50)
    plt.xticks([0, 24, 48])
    plt.tight_layout()
    plt.show()
