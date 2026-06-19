# gb_ig_explainer_main.py
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

from src.utils.seed_set import set_seed

from src.explainers.gb_ig import CTMPGIN_GBIGExplainer, compute_global_importance_on_loader
from src.explainers.stablity_report import (
    stability_report, 
    print_stability_report, 
    unstable_variables_report, 
    print_unstable_report_with_names,
    importance_mean_std_table,
    report
)
cur_dir = os.path.dirname(__file__)
model_path = os.path.join(cur_dir, '..', '..', 'runs', 'temp_ctmp_gin_ckpt', 'ctmp_epoch_36_loss_0.2738.pth')

def gb_ig_main(model, edge_index, dataset, test_loader, use_abs, device, reduce, use_mean_in_explain, seed, save_path, sample_ratio):
    explainer = CTMPGIN_GBIGExplainer(
        model=model,
        edge_index_vargraph=edge_index.detach().cpu(),
        ad_indices=dataset.col_info[2],  # type: ignore
        dis_indices=dataset.col_info[3], # type: ignore
        baseline_strategy="farthest",
        max_paths=1,            
        use_abs=use_abs,
        device=device,
    )
    
    rat = sample_ratio
    outs_var = []
    outs_ad = []
    outs_dis = []
    for s in [0, 1, 2]:
        set_seed(s)
        out = compute_global_importance_on_loader(
            explainer=explainer,
            model=model,
            dataloader=test_loader,
            edge_index=edge_index,
            device=device,
            sample_ratio=rat,
            seed=s,
            keep_all=False,
            reduce=reduce,
            use_mean_in_explain=use_mean_in_explain, # 
            verbose=True,
        )
        outs_var.append(out.global_importance_var.cpu().float())  # [N]
        outs_ad.append(out.global_importance_ad.cpu().float())
        outs_dis.append(out.global_importance_dis.cpu().float())
        
    col_names, col_dims, ad_col_index, dis_col_index = dataset.col_info

    col_names_ad = [col_names[i] for i in ad_col_index]
    col_names_dis = [col_names[i] for i in dis_col_index]

    # ---- after outs computed ----
    df_ms_var = importance_mean_std_table(outs_var, col_names_ad)
    df_ms_ad = importance_mean_std_table(outs_ad, col_names_ad)
    df_ms_dis = importance_mean_std_table(outs_dis, col_names_dis)

    report(df_ms_ad, outs_ad, col_names_ad, save_path, f"global_importance_ad_{seed}.csv")
    report(df_ms_dis, outs_dis, col_names_dis, save_path, f"global_importance_dis_{seed}.csv")
