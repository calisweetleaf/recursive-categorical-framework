Desktop\RCF-v2> python python_test/internal-contradictions-theory.py
# Internal Contradictions Theory Test Report

- Status: PASSED
- Timestamp (UTC): 2025-12-18T23:30:46Z
- Spec: `C:/Users/treyr/Desktop/RCF-v2/theoroms/ethics&stability/core stability frameworks/Internal_Contradictions_Theory.md`
- Spec SHA256: `d4ffb2907e6e7e9c20169328b2eb1bba67a71e3e712061fd5766cdfc7fb7be82`
- Display equations found: 21

## Equation Coverage

- Coverage: OK (validators scheduled: 19)

## Checks

- `equation_coverage`: OK (0.000s)
  - total_equations_found: `21`
  - unique_validators_scheduled: `19`
- `complexity_lower_bound_proxy`: OK (0.000s)
  - samples: `[{"n": 10, "T_M(n)": 20, "T(n)": 30, "ratio": 1.5}, {"n": 100, "T_M(n)": 200, "T(n)": 300, "ratio": 1.5}, {"n": 1000, "T_M(n)": 2000, "T(n)": 3000, "ratio": 1.5}]`
- `fisher_information_metric`: OK (0.000s)
  - sigma2: `2.0`
  - g: `0.5`
- `free_energy_identity`: OK (0.000s)
  - F: `2.1462041645277226`
  - KL: `0.8611997439356345`
  - -log_evidence: `1.2850044205920883`
  - abs_error: `4.440892098500626e-16`
  - posterior: `{"mu": 0.4666666666666666, "var": 0.3333333333333333}`
- `graph_path_product`: OK (0.000s)
  - path_products: `[0.5599999999999999, 0.3]`
  - normalized: `[0.6511627906976745, 0.34883720930232565]`
- `graph_pruning_constraint`: OK (0.000s)
  - epsilon: `0.2`
  - total_mi: `1.6`
  - picked_mi: `1.3`
  - picked_edges: `["a->b", "b->c"]`
- `hilbert_state_normalization`: OK (0.000s)
  - dim_psi: `3`
  - dim_phi: `2`
  - dim_tensor: `6`
  - norm2_tensor: `0.9999999999999999`
- `mdl_decomposition`: OK (0.000s)
  - n: `200`
  - mdl_model1: `423.7639122816739`
  - mdl_model2: `426.4045155126908`
  - chosen: `model1`
  - components: `{"L_M1": 2.649158683274018, "L_D_given_M1": 421.1147535983999, "L_M2": 5.298317366548036, "L_D_given_M2": 421.1061981461428}`  
- `mdl_simplicity_bias`: OK (0.000s)
  - n: `200`
  - mdl_model1: `423.7639122816739`
  - mdl_model2: `426.4045155126908`
  - chosen: `model1`
  - components: `{"L_M1": 2.649158683274018, "L_D_given_M1": 421.1147535983999, "L_M2": 5.298317366548036, "L_D_given_M2": 421.1061981461428}`  
- `momentum_update`: OK (0.000s)
  - loss_start: `0.5`
  - loss_end: `2.358584371918921e-07`
  - theta_end: `1.5005803248785614`
- `natural_gradient_update`: OK (0.000s)
  - theta_star: `-1.0`
  - theta_final: `-0.9999993367782408`
Desktop\RCF-v2\results\internal_contradictions_theory_test_report.md
Desktop\RCF-v2\results\internal_contradictions_theory_test_manifest.json