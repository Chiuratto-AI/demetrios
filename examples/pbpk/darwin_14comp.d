// Darwin PBPK 14-Compartment Model in Demetrios
// Author: Demetrios Chiuratto Agourakis

struct Drug {
    mw: f64,
    logp: f64,
    tpsa: f64,
    pka: f64,
    fu: f64,
    bp_ratio: f64,
    is_base: bool
}

struct Patient {
    weight: f64,
    age: f64
}

struct PBPKParams {
    cl_hepatic: f64,
    cl_renal: f64,
    vd: f64,
    ka: f64,
    f_oral: f64
}

fn get_blood_volume() -> f64 {
    return 5.0
}

fn get_liver_volume() -> f64 {
    return 1.8
}

fn get_kidney_volume() -> f64 {
    return 0.31
}

fn calculate_kp_liver(logp: f64, fu: f64) -> f64 {
    let base = 1.0
    let logp_contrib = logp * 0.3
    let sum = base + logp_contrib
    return sum / fu
}

fn calculate_kp_kidney(logp: f64, fu: f64) -> f64 {
    let base = 1.0
    let logp_contrib = logp * 0.25
    let sum = base + logp_contrib
    return sum / fu
}

fn calculate_kp_adipose(logp: f64, fu: f64) -> f64 {
    let base = 0.5
    let logp_contrib = logp * 0.8
    let sum = base + logp_contrib
    return sum / fu
}

fn validate_vd(vd: f64) -> bool {
    if vd > 0.0 {
        if vd < 2000.0 {
            return true
        }
    }
    return false
}

fn validate_cl(cl: f64) -> bool {
    if cl > 0.0 {
        if cl < 5000.0 {
            return true
        }
    }
    return false
}

fn calculate_cmax(dose: f64, vd: f64, f: f64, bp: f64) -> f64 {
    let amt = dose * f
    let c_blood = amt / vd
    let c_plasma = c_blood / bp
    let c_ng = c_plasma * 1000.0
    return c_ng
}

fn calculate_auc(dose: f64, cl: f64, f: f64) -> f64 {
    let amt = dose * f
    let auc_ug = amt / cl
    let auc_ng = auc_ug * 1000.0
    return auc_ng
}

fn calculate_half_life(vd: f64, cl: f64) -> f64 {
    let ratio = vd / cl
    let t_half = 0.693 * ratio
    return t_half
}

fn is_within_2fold(pred: f64, obs: f64) -> bool {
    let ratio = pred / obs
    if ratio >= 0.5 {
        if ratio <= 2.0 {
            return true
        }
    }
    return false
}

fn scale_volume(ref_vol: f64, weight: f64) -> f64 {
    let ratio = weight / 70.0
    return ref_vol * ratio
}

fn main() -> i32 {
    let drug = Drug {
        mw: 325.77,
        logp: 2.5,
        tpsa: 30.2,
        pka: 5.2,
        fu: 0.04,
        bp_ratio: 0.66,
        is_base: true
    }
    
    let patient = Patient {
        weight: 70.0,
        age: 35.0
    }
    
    let params = PBPKParams {
        cl_hepatic: 27.0,
        cl_renal: 0.5,
        vd: 77.0,
        ka: 4.0,
        f_oral: 0.44
    }
    
    let vd_ok = validate_vd(params.vd)
    let cl_ok = validate_cl(params.cl_hepatic)
    
    let kp_liver = calculate_kp_liver(drug.logp, drug.fu)
    let kp_kidney = calculate_kp_kidney(drug.logp, drug.fu)
    let kp_adipose = calculate_kp_adipose(drug.logp, drug.fu)
    
    let dose = 7.5
    let cl_total = params.cl_hepatic + params.cl_renal
    
    let cmax = calculate_cmax(dose, params.vd, params.f_oral, drug.bp_ratio)
    let auc = calculate_auc(dose, cl_total, params.f_oral)
    let t_half = calculate_half_life(params.vd, cl_total)
    
    let cmax_obs = 32.5
    let auc_obs = 89.3
    
    let cmax_ok = is_within_2fold(cmax, cmax_obs)
    let auc_ok = is_within_2fold(auc, auc_obs)
    
    return 0
}
