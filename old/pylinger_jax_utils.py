def get_grho_and_adotrad(H0, Tcmb):
    grhom = 3.3379e-11 * H0 ** 2
    grhog = 1.4952e-13 * Tcmb ** 4
    grhor = 3.3957e-14 * Tcmb ** 4
    adotrad = 2.8948e-7 * Tcmb ** 2
    return grhom, grhog, grhor, adotrad