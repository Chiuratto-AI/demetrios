// DARWIN PBPK - COMPLETE ODE (Unit-Safe)
// Demetrios Language

struct Drug {
    mw: f64,
    logp: f64,
    fu: f64,
    bp_ratio: f64,
    pka: f64,
    is_base: bool
}

struct Volumes {
    v_blood: f64@L,
    v_liver: f64@L,
    v_kidney: f64@L,
    v_gut: f64@L,
    v_muscle: f64@L,
    v_adipose: f64@L,
    v_rest: f64@L
}

struct Flows {
    q_liver: f64@L_per_h,
    q_kidney: f64@L_per_h,
    q_gut: f64@L_per_h,
    q_muscle: f64@L_per_h,
    q_adipose: f64@L_per_h,
    q_rest: f64@L_per_h
}

struct Kp {
    kp_liver: f64,
    kp_kidney: f64,
    kp_gut: f64,
    kp_muscle: f64,
    kp_adipose: f64,
    kp_rest: f64
}

struct State {
    c_venous: f64,
    c_liver: f64,
    c_kidney: f64,
    c_gut: f64,
    c_muscle: f64,
    c_adipose: f64,
    c_rest: f64,
    a_gut_lumen: f64
}

struct Deriv {
    dc_venous: f64,
    dc_liver: f64,
    dc_kidney: f64,
    dc_gut: f64,
    dc_muscle: f64,
    dc_adipose: f64,
    dc_rest: f64,
    da_gut: f64
}

fn create_volumes() -> Volumes {
    return Volumes {
        v_blood: 5.2,
        v_liver: 1.69,
        v_kidney: 0.28,
        v_gut: 1.18,
        v_muscle: 28.0,
        v_adipose: 14.0,
        v_rest: 10.0
    }
}

fn create_flows() -> Flows {
    return Flows {
        q_liver: 81.9,
        q_kidney: 55.8,
        q_gut: 57.6,
        q_muscle: 42.0,
        q_adipose: 18.6,
        q_rest: 50.0
    }
}

fn calc_kp(logp: f64, fu: f64, is_base: bool, f_lipid: f64) -> f64 {
    let kp_base = (0.7 + f_lipid * logp * 0.5) / fu
    let kp = if is_base { kp_base * 1.3 } else { kp_base }
    if kp < 0.5 { return 0.5 }
    if kp > 50.0 { return 50.0 }
    return kp
}

fn calc_all_kp(drug: Drug) -> Kp {
    return Kp {
        kp_liver: calc_kp(drug.logp, drug.fu, drug.is_base, 0.05),
        kp_kidney: calc_kp(drug.logp, drug.fu, drug.is_base, 0.03),
        kp_gut: calc_kp(drug.logp, drug.fu, drug.is_base, 0.04),
        kp_muscle: calc_kp(drug.logp, drug.fu, drug.is_base, 0.02),
        kp_adipose: calc_kp(drug.logp, drug.fu, drug.is_base, 0.80),
        kp_rest: calc_kp(drug.logp, drug.fu, drug.is_base, 0.05)
    }
}

fn compute_deriv(s: State, v: Volumes, q: Flows, kp: Kp, cl_h: f64, cl_r: f64, fu: f64, ka: f64) -> Deriv {
    let c_art = s.c_venous
    
    let dc_liver = (q.q_liver / v.v_liver) * (c_art - s.c_liver / kp.kp_liver) - (cl_h / v.v_liver) * s.c_liver * fu
    let dc_kidney = (q.q_kidney / v.v_kidney) * (c_art - s.c_kidney / kp.kp_kidney) - (cl_r / v.v_kidney) * s.c_kidney * fu
    let dc_gut = (q.q_gut / v.v_gut) * (c_art - s.c_gut / kp.kp_gut) + (ka / v.v_gut) * s.a_gut_lumen
    let dc_muscle = (q.q_muscle / v.v_muscle) * (c_art - s.c_muscle / kp.kp_muscle)
    let dc_adipose = (q.q_adipose / v.v_adipose) * (c_art - s.c_adipose / kp.kp_adipose)
    let dc_rest = (q.q_rest / v.v_rest) * (c_art - s.c_rest / kp.kp_rest)
    
    let q_total = q.q_liver + q.q_kidney + q.q_muscle + q.q_adipose + q.q_rest
    let ven_ret = q.q_liver * s.c_liver / kp.kp_liver + q.q_kidney * s.c_kidney / kp.kp_kidney + q.q_gut * s.c_gut / kp.kp_gut + q.q_muscle * s.c_muscle / kp.kp_muscle + q.q_adipose * s.c_adipose / kp.kp_adipose + q.q_rest * s.c_rest / kp.kp_rest
    let dc_venous = (ven_ret - q_total * s.c_venous) / v.v_blood
    
    let da_gut = 0.0 - ka * s.a_gut_lumen
    
    return Deriv {
        dc_venous: dc_venous,
        dc_liver: dc_liver,
        dc_kidney: dc_kidney,
        dc_gut: dc_gut,
        dc_muscle: dc_muscle,
        dc_adipose: dc_adipose,
        dc_rest: dc_rest,
        da_gut: da_gut
    }
}

fn add_state(s: State, d: Deriv, dt: f64) -> State {
    return State {
        c_venous: s.c_venous + d.dc_venous * dt,
        c_liver: s.c_liver + d.dc_liver * dt,
        c_kidney: s.c_kidney + d.dc_kidney * dt,
        c_gut: s.c_gut + d.dc_gut * dt,
        c_muscle: s.c_muscle + d.dc_muscle * dt,
        c_adipose: s.c_adipose + d.dc_adipose * dt,
        c_rest: s.c_rest + d.dc_rest * dt,
        a_gut_lumen: s.a_gut_lumen + d.da_gut * dt
    }
}

fn scale_d(d: Deriv, s: f64) -> Deriv {
    return Deriv {
        dc_venous: d.dc_venous * s,
        dc_liver: d.dc_liver * s,
        dc_kidney: d.dc_kidney * s,
        dc_gut: d.dc_gut * s,
        dc_muscle: d.dc_muscle * s,
        dc_adipose: d.dc_adipose * s,
        dc_rest: d.dc_rest * s,
        da_gut: d.da_gut * s
    }
}

fn add_d(d1: Deriv, d2: Deriv) -> Deriv {
    return Deriv {
        dc_venous: d1.dc_venous + d2.dc_venous,
        dc_liver: d1.dc_liver + d2.dc_liver,
        dc_kidney: d1.dc_kidney + d2.dc_kidney,
        dc_gut: d1.dc_gut + d2.dc_gut,
        dc_muscle: d1.dc_muscle + d2.dc_muscle,
        dc_adipose: d1.dc_adipose + d2.dc_adipose,
        dc_rest: d1.dc_rest + d2.dc_rest,
        da_gut: d1.da_gut + d2.da_gut
    }
}

fn rk4(s: State, v: Volumes, q: Flows, kp: Kp, cl_h: f64, cl_r: f64, fu: f64, ka: f64, dt: f64) -> State {
    let k1 = compute_deriv(s, v, q, kp, cl_h, cl_r, fu, ka)
    let s2 = add_state(s, k1, dt * 0.5)
    let k2 = compute_deriv(s2, v, q, kp, cl_h, cl_r, fu, ka)
    let s3 = add_state(s, k2, dt * 0.5)
    let k3 = compute_deriv(s3, v, q, kp, cl_h, cl_r, fu, ka)
    let s4 = add_state(s, k3, dt)
    let k4 = compute_deriv(s4, v, q, kp, cl_h, cl_r, fu, ka)
    
    let k2x = scale_d(k2, 2.0)
    let k3x = scale_d(k3, 2.0)
    let sum1 = add_d(k1, k2x)
    let sum2 = add_d(sum1, k3x)
    let sum3 = add_d(sum2, k4)
    let kavg = scale_d(sum3, 0.1666667)
    
    return add_state(s, kavg, dt)
}

fn iv_state(dose: f64, v_blood: f64) -> State {
    let c0 = dose / v_blood
    return State {
        c_venous: c0,
        c_liver: 0.0,
        c_kidney: 0.0,
        c_gut: 0.0,
        c_muscle: 0.0,
        c_adipose: 0.0,
        c_rest: 0.0,
        a_gut_lumen: 0.0
    }
}

fn oral_state(dose: f64, f_oral: f64) -> State {
    return State {
        c_venous: 0.0,
        c_liver: 0.0,
        c_kidney: 0.0,
        c_gut: 0.0,
        c_muscle: 0.0,
        c_adipose: 0.0,
        c_rest: 0.0,
        a_gut_lumen: dose * f_oral
    }
}

fn max_f(a: f64, b: f64) -> f64 {
    if a > b { return a }
    return b
}

fn fold_err(pred: f64, obs: f64) -> f64 {
    let r = pred / obs
    if r >= 1.0 { return r }
    return obs / pred
}

fn main() -> i32 {
    println("================================================")
    println("  DARWIN PBPK - FULL ODE + RK4")
    println("  Demetrios Language - 7 Compartments")
    println("================================================")
    println("")
    
    let v = create_volumes()
    let q = create_flows()
    
    // MIDAZOLAM IV
    println("1. MIDAZOLAM 2mg IV")
    let midaz = Drug { mw: 325.8, logp: 3.89, fu: 0.03, bp_ratio: 0.64, pka: 6.15, is_base: true }
    let kp = calc_all_kp(midaz)
    
    let dose1 = 2.0
    let s0 = iv_state(dose1, 5.2)
    let cl_h = 27.0
    let cl_r = 0.5
    let ka = 4.0
    let dt = 0.05
    
    println("  Initial C_venous:")
    println(s0.c_venous)
    
    let s1 = rk4(s0, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s2 = rk4(s1, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s3 = rk4(s2, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s4 = rk4(s3, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s5 = rk4(s4, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s6 = rk4(s5, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s7 = rk4(s6, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s8 = rk4(s7, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s9 = rk4(s8, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s10 = rk4(s9, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    
    println("  After 0.5h C_venous:")
    println(s10.c_venous)
    println("  After 0.5h C_liver:")
    println(s10.c_liver)
    
    let s20 = rk4(s10, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    let s20 = rk4(s20, v, q, kp, cl_h, cl_r, midaz.fu, ka, dt)
    
    println("  After 1h C_venous:")
    println(s20.c_venous)
    
    let cmax_pred = s0.c_venous
    let cmax_obs = 0.039
    let fe1 = fold_err(cmax_pred, cmax_obs)
    println("  Cmax obs: 0.039")
    println("  FE:")
    println(fe1)
    println("")
    
    // CAFFEINE ORAL
    println("2. CAFFEINE 200mg ORAL")
    let caff = Drug { mw: 194.2, logp: 0.0, fu: 0.64, bp_ratio: 0.89, pka: 10.4, is_base: false }
    let kp2 = calc_all_kp(caff)
    
    let dose2 = 200.0
    let f_oral = 0.99
    let c0 = oral_state(dose2, f_oral)
    let cl_h2 = 6.0
    let cl_r2 = 0.1
    let ka2 = 4.5
    
    println("  Initial A_gut_lumen:")
    println(c0.a_gut_lumen)
    
    let c1 = rk4(c0, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c2 = rk4(c1, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c3 = rk4(c2, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c4 = rk4(c3, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c5 = rk4(c4, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c6 = rk4(c5, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c7 = rk4(c6, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c8 = rk4(c7, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c9 = rk4(c8, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c10 = rk4(c9, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    
    println("  After 0.5h C_venous:")
    println(c10.c_venous)
    println("  After 0.5h A_gut_lumen:")
    println(c10.a_gut_lumen)
    
    let c20 = rk4(c10, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    let c20 = rk4(c20, v, q, kp2, cl_h2, cl_r2, caff.fu, ka2, dt)
    
    println("  After 1h C_venous:")
    println(c20.c_venous)
    
    let cmax2 = max_f(c10.c_venous, c20.c_venous)
    let cmax2_obs = 5.0
    let fe2 = fold_err(cmax2, cmax2_obs)
    println("  Cmax pred:")
    println(cmax2)
    println("  Cmax obs: 5.0")
    println("  FE:")
    println(fe2)
    println("")
    
    println("================================================")
    println("  ODE + RK4 WORKING IN DEMETRIOS!")
    println("================================================")
    
    return 0
}
