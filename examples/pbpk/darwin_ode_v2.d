// DARWIN PBPK - ODE + RK4 FIXED
// Demetrios Language

struct State {
    cv: f64,
    cl: f64,
    ck: f64,
    cg: f64,
    cm: f64,
    ca: f64,
    ag: f64
}

struct Deriv {
    dv: f64,
    dl: f64,
    dk: f64,
    dg: f64,
    dm: f64,
    da: f64,
    dag: f64
}

fn kp_calc(logp: f64, fu: f64, is_base: bool) -> f64 {
    let base = 0.7 / fu
    if logp > 0.0 {
        let base = (0.7 + logp * 0.1) / fu
    }
    let kp = if is_base { base * 1.2 } else { base }
    if kp < 0.5 { return 0.5 }
    if kp > 30.0 { return 30.0 }
    return kp
}

fn ode_iv(s: State, vb: f64, vl: f64, vk: f64, vm: f64, va: f64, ql: f64, qk: f64, qm: f64, qa: f64, kl: f64, kk: f64, km: f64, ka_kp: f64, clh: f64, clr: f64, fu: f64) -> Deriv {
    let dl = (ql / vl) * (s.cv - s.cl / kl) - (clh * fu / vl) * s.cl
    let dk = (qk / vk) * (s.cv - s.ck / kk) - (clr * fu / vk) * s.ck
    let dm = (qm / vm) * (s.cv - s.cm / km)
    let da = (qa / va) * (s.cv - s.ca / ka_kp)
    
    let qtot = ql + qk + qm + qa
    let vret = ql * s.cl / kl + qk * s.ck / kk + qm * s.cm / km + qa * s.ca / ka_kp
    let dv = (vret - qtot * s.cv) / vb
    
    return Deriv { dv: dv, dl: dl, dk: dk, dg: 0.0, dm: dm, da: da, dag: 0.0 }
}

fn ode_oral(s: State, vb: f64, vl: f64, vk: f64, vg: f64, vm: f64, va: f64, ql: f64, qk: f64, qg: f64, qm: f64, qa: f64, kl: f64, kk: f64, kg: f64, km: f64, ka_kp: f64, clh: f64, clr: f64, fu: f64, ka: f64) -> Deriv {
    let abs_rate = ka * s.ag
    
    let dl = (ql / vl) * (s.cv - s.cl / kl) - (clh * fu / vl) * s.cl
    let dk = (qk / vk) * (s.cv - s.ck / kk) - (clr * fu / vk) * s.ck
    let dg = (qg / vg) * (s.cv - s.cg / kg) + abs_rate / vg
    let dm = (qm / vm) * (s.cv - s.cm / km)
    let da = (qa / va) * (s.cv - s.ca / ka_kp)
    
    let qtot = ql + qk + qm + qa
    let vret = ql * s.cl / kl + qk * s.ck / kk + qg * s.cg / kg + qm * s.cm / km + qa * s.ca / ka_kp
    let dv = (vret - qtot * s.cv) / vb
    
    let dag = 0.0 - ka * s.ag
    
    return Deriv { dv: dv, dl: dl, dk: dk, dg: dg, dm: dm, da: da, dag: dag }
}

fn add_s(s: State, d: Deriv, h: f64) -> State {
    return State {
        cv: s.cv + d.dv * h,
        cl: s.cl + d.dl * h,
        ck: s.ck + d.dk * h,
        cg: s.cg + d.dg * h,
        cm: s.cm + d.dm * h,
        ca: s.ca + d.da * h,
        ag: s.ag + d.dag * h
    }
}

fn scale_d(d: Deriv, f: f64) -> Deriv {
    return Deriv { dv: d.dv * f, dl: d.dl * f, dk: d.dk * f, dg: d.dg * f, dm: d.dm * f, da: d.da * f, dag: d.dag * f }
}

fn add_d(a: Deriv, b: Deriv) -> Deriv {
    return Deriv { dv: a.dv + b.dv, dl: a.dl + b.dl, dk: a.dk + b.dk, dg: a.dg + b.dg, dm: a.dm + b.dm, da: a.da + b.da, dag: a.dag + b.dag }
}

fn rk4_iv(s: State, vb: f64, vl: f64, vk: f64, vm: f64, va: f64, ql: f64, qk: f64, qm: f64, qa: f64, kl: f64, kk: f64, km: f64, ka_kp: f64, clh: f64, clr: f64, fu: f64, dt: f64) -> State {
    let k1 = ode_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, clh, clr, fu)
    let s2 = add_s(s, k1, dt * 0.5)
    let k2 = ode_iv(s2, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, clh, clr, fu)
    let s3 = add_s(s, k2, dt * 0.5)
    let k3 = ode_iv(s3, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, clh, clr, fu)
    let s4 = add_s(s, k3, dt)
    let k4 = ode_iv(s4, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, clh, clr, fu)
    
    let t1 = add_d(k1, scale_d(k2, 2.0))
    let t2 = add_d(t1, scale_d(k3, 2.0))
    let t3 = add_d(t2, k4)
    let kavg = scale_d(t3, 0.1666667)
    return add_s(s, kavg, dt)
}

fn rk4_oral(s: State, vb: f64, vl: f64, vk: f64, vg: f64, vm: f64, va: f64, ql: f64, qk: f64, qg: f64, qm: f64, qa: f64, kl: f64, kk: f64, kg: f64, km: f64, ka_kp: f64, clh: f64, clr: f64, fu: f64, ka: f64, dt: f64) -> State {
    let k1 = ode_oral(s, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, kl, kk, kg, km, ka_kp, clh, clr, fu, ka)
    let s2 = add_s(s, k1, dt * 0.5)
    let k2 = ode_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, kl, kk, kg, km, ka_kp, clh, clr, fu, ka)
    let s3 = add_s(s, k2, dt * 0.5)
    let k3 = ode_oral(s3, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, kl, kk, kg, km, ka_kp, clh, clr, fu, ka)
    let s4 = add_s(s, k3, dt)
    let k4 = ode_oral(s4, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, kl, kk, kg, km, ka_kp, clh, clr, fu, ka)
    
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
    println("  Demetrios - 6 Compartment")
    println("================================================")
    println("")
    
    let vb = 5.2
    let vl = 1.69
    let vk = 0.28
    let vg = 1.18
    let vm = 28.0
    let va = 14.0
    
    let ql = 25.0
    let qk = 20.0
    let qg = 15.0
    let qm = 15.0
    let qa = 5.0
    
    // 1. MIDAZOLAM IV
    println("1. MIDAZOLAM 2mg IV")
    let kl = kp_calc(3.89, 0.03, true)
    let kk = kp_calc(3.89, 0.03, true)
    let km = kp_calc(3.89, 0.03, true)
    let ka_kp = kp_calc(3.89, 0.03, true)
    
    let dose1 = 2.0
    let c0 = dose1 / vb
    let s = State { cv: c0, cl: 0.0, ck: 0.0, cg: 0.0, cm: 0.0, ca: 0.0, ag: 0.0 }
    let dt = 0.01
    
    println("  C0:")
    println(c0)
    
    let cmax = c0
    
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    let s = rk4_iv(s, vb, vl, vk, vm, va, ql, qk, qm, qa, kl, kk, km, ka_kp, 27.0, 0.5, 0.03, dt)
    
    println("  t=0.1h Cv:")
    println(s.cv)
    
    let fe1 = fe(cmax, 0.039)
    println("  FE:")
    println(fe1)
    println("")
    
    // 2. CAFFEINE ORAL
    println("2. CAFFEINE 200mg ORAL")
    let c_kl = kp_calc(0.0, 0.64, false)
    let c_kk = kp_calc(0.0, 0.64, false)
    let c_kg = kp_calc(0.0, 0.64, false)
    let c_km = kp_calc(0.0, 0.64, false)
    let c_ka_kp = kp_calc(0.0, 0.64, false)
    
    let dose2 = 200.0
    let s2 = State { cv: 0.0, cl: 0.0, ck: 0.0, cg: 0.0, cm: 0.0, ca: 0.0, ag: dose2 * 0.99 }
    
    println("  A_gut0:")
    println(s2.ag)
    
    let cmax2 = 0.0
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    let s2 = rk4_oral(s2, vb, vl, vk, vg, vm, va, ql, qk, qg, qm, qa, c_kl, c_kk, c_kg, c_km, c_ka_kp, 6.0, 0.1, 0.64, 2.0, dt)
    let cmax2 = max2(cmax2, s2.cv)
    
    println("  t=0.1h Cv:")
    println(s2.cv)
    println("  A_gut remaining:")
    println(s2.ag)
    println("  Cmax:")
    println(cmax2)
    
    let fe2 = fe(cmax2, 5.0)
    println("  FE:")
    println(fe2)
    println("")
    
    println("================================================")
    println("  SUCCESS! ODE + RK4 IN DEMETRIOS")
    println("================================================")
    
    return 0
}
