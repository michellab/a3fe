from .autocorrelation import get_statistical_inefficiency
from .compare import compare_variances_brown_forsythe, get_comparitive_convergence_data
from .detect_equil import (
    check_equil_block_gradient,
    check_equil_multiwindow_modified_geweke,
    check_equil_multiwindow_paired_t,
    dummy_check_equil_multiwindow,
    get_gelman_rubin_rhat,
)
from .mbar import run_mbar
from .plot import (
    general_plot,
    p_plot,
    plot_against_exp,
    plot_av_waters,
    plot_comparitive_convergence,
    plot_comparitive_convergence_on_ax,
    plot_comparitive_convergence_sem,
    plot_comparitive_convergence_sem_on_ax,
    plot_equilibration_time,
    plot_gelman_rubin_rhat,
    plot_gradient_hists,
    plot_gradient_stats,
    plot_gradient_timeseries,
    plot_mbar_pmf,
    plot_normality,
    plot_overlap_mat,
    plot_overlap_mats,
)
from .process_grads import GradientData
from .rmsd import get_rmsd
from .waters import (
    get_av_waters_lambda_window,
    get_av_waters_simulation,
    get_av_waters_stage,
)
