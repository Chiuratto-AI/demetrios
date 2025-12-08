// DARWIN PBPK - ODE + RK4 (Dimensionless Internal)
// Demetrios Language

struct Drug {
    mw: f64,
    logp: f64,
    fu: f64,
    bp: f64,
    is_base: bool
}

struct State {
    cv: f64,
    cl: f64,
    ck: f64,
    cg: f64,
    cm: f64,
    ca: f64,
    cr: f64,
    ag: f64
}

struct Deriv {
    dv: f64,
    dl: f64,
    dk: f64,
    dg: f64,
    dm: f64,
    da: f64,
    dr: f64,
    dag: f64
}

fn kp_calc(logp: f64, fu: f64, is_base: bool, fl: f64) -> f64 {
    let base = (0.7 + fl * logp * 0.5) / fu
    let kp = if is_base { base * 1.3 } else { base }
    if kp < 0.5 { return 0.5 }
    if kp > 50.0 { return 50.0 }
    return kp
}

fn ode(s: State, vb: f64, vl: f64, vk: f64, vg: f64, vm: f64, va: f64, vr: f64, ql: f64, qk: f64, qg: f64, qm: f64, qa: f64, qr: f64, kl: f64, kk: f64, kg: f64, km: f64, ka_kp: f64, kr: f64, clh: f64, clr: f64, fu: f64, ka: f64) -> Deriv {
    let cart = s.cv
    
    let dl = (ql / vl) * (cart - s.cl / kl) - (clh / vl) * s.cl * fu
    let dk = (qk / vk) * (cart - s.ck / kk) - (clr / vk) * s.ck * fu
    let dg = (qg / vg) * (cart - s.cg / kg) + (ka / vg) * s.ag
    let dm = (qm / vm) * (cart - s.cm / km)
    let da = (qa / va) * (cart - s.ca / ka_kp)
    let dr = (qr / vr) * (cart - s.cr / kr)
    
    let qtot = ql + qk + qm + qa + qr
    let vret = ql * s.cl / kl + qk * s.ck / kk + qg * s.cg / kg + qm * s.cm / km + qa * s.ca / ka_kp + qr * s.cr / kr
    let dv = (vret - qtot * s.cv) / vb
    
    let dag = 0.0 - ka * s.ag
    
    return Deriv { dv: dv, dl: dl, dk: dk, dg: dg, dm: dm, da: da, dr: dr, dag: dag }
}

fn add_s(s: State, d: Deriv, h: f64) -> State {
    return State {
        cv: s.cv + d.dv * h,
        cl: s.cl + d.dl * h,
        ck: s.ck + d.dk * h,
        cg: s.cg + d.dg * h,
        cm: s.cm + d.dm * h,
        ca: s.ca + d.da * h,
        cr: s.cr + d.dr * h,
        ag: s.ag + d.dag * h
    }
}

fn scale_d(d: Deriv, f: f64) -> Deriv {
    return Deriv {
        dv: d.dv * f, dl: d.dl * f, dk: d.dk * f, dg: d.dg * f,
        dm: d.dm * f, da: d.da * f, dr: d.dr * f, dag: d.dag * f
    }
}

fn add_d(a: Deriv, b: Deriv) -> Deriv {
    return Deriv {
        dv: a.dv + b.dv, dl: a.dl + b.dl, dk: a.dk + b.dk, dg: a.dg + b.dg,
        dm: a.dm + b.dm, da: a.da + b.da, dr: a.dr + b.dr, dag: a.dag + b.dag
    }
}

fn rk4(s: State, vb: f64, vl: f64, vk: f64, vg: f64, vm: f64, va: f64, vr: f64, ql: f64, qk: f64, qg: f64, qm: f64, qa: f64, qr: f64, kl: f64, kk: f64, kg: f64, km: f64, ka_kp: f64, kr: f64, clh: f64, clr: f64, fu: f64, ka: f64, dt: f64) -> State {
    let k1 = ode(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, fu, ka)
    let s2 = add_s(s, k1, dt * 0.5)
    let k2 = ode(s2, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, fu, ka)
    let s3 = add_s(s, k2, dt * 0.5)
    let k3 = ode(s3, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, fu, ka)
    let s4 = add_s(s, k3, dt)
    let k4 = ode(s4, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, fu, ka)
    
    let t1 = add_d(k1, scale_d(k2, 2.0))
    let t2 = add_d(t1, scale_d(k3, 2.0))
    let t3 = add_d(t2, k4)
    let kavg = scale_d(t3, 0.1666667)
    
    return add_s(s, kavg, dt)
}

fn fe(p: f64, o: f64) -> f64 {
    let r = p / o
    if r >= 1.0 { return r }
    return o / p
}

fn max2(a: f64, b: f64) -> f64 {
    if a > b { return a }
    return b
}

fn main() -> i32 {
    println("================================================")
    println("  DARWIN PBPK - ODE + RK4")
    println("  7-Compartment Model")
    println("================================================")
    println("")
    
    let vb = 5.2
    let vl = 1.69
    let vk = 0.28
    let vg = 1.18
    let vm = 28.0
    let va = 14.0
    let vr = 10.0
    
    let ql = 81.9
    let qk = 55.8
    let qg = 57.6
    let qm = 42.0
    let qa = 18.6
    let qr = 50.0
    
    // MIDAZOLAM
    println("1. MIDAZOLAM 2mg IV")
    let m_fu = 0.03
    let m_logp = 3.89
    let kl = kp_calc(m_logp, m_fu, true, 0.05)
    let kk = kp_calc(m_logp, m_fu, true, 0.03)
    let kg = kp_calc(m_logp, m_fu, true, 0.04)
    let km = kp_calc(m_logp, m_fu, true, 0.02)
    let ka_kp = kp_calc(m_logp, m_fu, true, 0.80)
    let kr = kp_calc(m_logp, m_fu, true, 0.05)
    
    let dose1 = 2.0
    let c0 = dose1 / vb
    let s0 = State { cv: c0, cl: 0.0, ck: 0.0, cg: 0.0, cm: 0.0, ca: 0.0, cr: 0.0, ag: 0.0 }
    let clh = 27.0
    let clr = 0.5
    let ka = 0.0
    let dt = 0.05
    
    println("  C0:")
    println(c0)
    
    let s = rk4(s0, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    
    println("  t=0.5h Cv:")
    println(s.cv)
    println("  t=0.5h Cl:")
    println(s.cl)
    
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    let s = rk4(s, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, kl, kk, kg, km, ka_kp, kr, clh, clr, m_fu, ka, dt)
    
    println("  t=1h Cv:")
    println(s.cv)
    
    let fe1 = fe(c0, 0.039)
    println("  FE (Cmax):")
    println(fe1)
    println("")
    
    // CAFFEINE ORAL
    println("2. CAFFEINE 200mg ORAL")
    let c_fu = 0.64
    let c_logp = 0.0
    let c_kl = kp_calc(c_logp, c_fu, false, 0.05)
    let c_kk = kp_calc(c_logp, c_fu, false, 0.03)
    let c_kg = kp_calc(c_logp, c_fu, false, 0.04)
    let c_km = kp_calc(c_logp, c_fu, false, 0.02)
    let c_ka_kp = kp_calc(c_logp, c_fu, false, 0.80)
    let c_kr = kp_calc(c_logp, c_fu, false, 0.05)
    
    let dose2 = 200.0
    let f_oral = 0.99
    let c0_oral = State { cv: 0.0, cl: 0.0, ck: 0.0, cg: 0.0, cm: 0.0, ca: 0.0, cr: 0.0, ag: dose2 * f_oral }
    let c_clh = 6.0
    let c_clr = 0.1
    let c_ka = 4.5
    
    println("  A_gut0:")
    println(c0_oral.ag)
    
    let c = rk4(c0_oral, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = c.cv
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    
    println("  t=0.5h Cv:")
    println(c.cv)
    println("  Cmax so far:")
    println(cmax)
    
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    let c = rk4(c, vb, vl, vk, vg, vm, va, vr, ql, qk, qg, qm, qa, qr, c_kl, c_kk, c_kg, c_km, c_ka_kp, c_kr, c_clh, c_clr, c_fu, c_ka, dt)
    let cmax = max2(cmax, c.cv)
    
    println("  t=1h Cv:")
    println(c.cv)
    println("  Cmax final:")
    println(cmax)
    
    let fe2 = fe(cmax, 5.0)
    println("  FE (Cmax):")
    println(fe2)
    println("")
    
    println("================================================")
    println("  ODE + RK4 WORKING IN DEMETRIOS!")
    println("================================================")
    
    return 0
}
