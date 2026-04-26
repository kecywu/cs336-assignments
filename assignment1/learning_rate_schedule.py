import math 

def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):

    if t < Tw:
        return t / Tw * alpha_max
    elif t >= Tw and t <= Tc:
        return alpha_min + 0.5 * (1 + math.cos((t-Tw)/(Tc-Tw)*math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min