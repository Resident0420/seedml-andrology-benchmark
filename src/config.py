# ============================================
# Global Configuration
# ============================================

TARGET = "diagnostic"
RANDOM_STATE = 42


# ============================================
# Tier 1 Features
# ============================================

TIER1_FEATURES = [
    "sample_state",
    "sample_appearance",
    "sample_agglutination",
    "sample_viscosity",
    "sample_liquefaction",
    "sample_red_blood_cells",
    "sample_leukocytes",
    "sample_ph",
    "sample_cells_round",
    "sample_vitality",
    "sample_survival_test",
    "sample_vol_initial",
    "sample_concentration_initial",
    "sample_morpho_normal",
    "sample_morpho_kruger",
    "sample_production_total",
    "sample_num_prog_mob_total",
    "sample_bodies_gelatinous",
]


# ============================================
# Extended Feature Groups
# ============================================

MORPHOLOGY_PRE = [
    "sample_num_spz_counted_pre",
    "sample_num_spz_normal_pre",
    "sample_morphology_normal_pre",
    "sample_heads_total_pre",
    "sample_heads_elongated_pre",
    "sample_heads_piriform_pre",
    "sample_heads_round_pre",
    "sample_heads_amorphous_pre",
    "sample_heads_macrocephalus_pre",
    "sample_heads_microcephalus_pre",
    "sample_heads_vacuole_pre",
    "sample_heads_small_acrosome_pre",
    "sample_heads_double_pre",
    "sample_heads_combined_pre",
    "sample_necks_bent_pre",
    "sample_necks_tails_total_pre",
    "sample_necks_ins_asymmetric_pre",
    "sample_necks_thick_pre",
    "sample_necks_thin_pre",
    "sample_necks_combined_pre",
    "sample_tails_short_pre",
    "sample_tails_broken_pre",
    "sample_tails_rolled_pre",
    "sample_tails_multiple_pre",
    "sample_tails_combined_pre",
    "sample_anormal_total_pre",
]

BIOMARKERS = [
    "sample_scd",
    "sample_citric",
    "sample_fructose",
    "sample_spz_swollen",
]


# ============================================
# Benchmark Feature Sets
# ============================================

TIER2_FEATURES = TIER1_FEATURES + MORPHOLOGY_PRE
TIER3_FEATURES = TIER2_FEATURES + BIOMARKERS


# ============================================
# Tier 1 Variable Types
# ============================================

NUMERIC_FEATURES = [
    "sample_leukocytes",
    "sample_ph",
    "sample_cells_round",
    "sample_vitality",
    "sample_survival_test",
    "sample_vol_initial",
    "sample_concentration_initial",
    "sample_morpho_normal",
    "sample_morpho_kruger",
    "sample_production_total",
    "sample_num_prog_mob_total",
]

CATEGORICAL_FEATURES = [
    "sample_state",
    "sample_appearance",
    "sample_agglutination",
    "sample_viscosity",
    "sample_liquefaction",
    "sample_red_blood_cells",
    "sample_bodies_gelatinous",
]


# ============================================
# Configuration Validation
# ============================================

assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)

assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(TIER1_FEATURES)