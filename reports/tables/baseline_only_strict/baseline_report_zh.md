# Baseline-only reward/cost report

本报告只包含三个文献 baseline：`kh_dual_control`、`arcari_dual_smpc`、`tv_gp_lcb`。不包含 oracle，也不报告 oracle regret。

指标定义：`net_reward = - total_cost`；`acc_task_reward = - task/state cost`；action cost 分为 energy 与 switch。

## Scalar summary
| environment | twin_gap | regime | baseline | n | mean_net_reward | stderr_net_reward | mean_total_cost | stderr_total_cost | mean_acc_task_reward | stderr_acc_task_reward | mean_acc_energy_cost | stderr_acc_energy_cost | mean_acc_switch_cost | stderr_acc_switch_cost | mean_acc_failure_cost | stderr_acc_failure_cost | mean_terminal_cost | stderr_terminal_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scalar | gap | drifting | arcari_dual_smpc | 3 | -15.18 | 0.9014 | 15.18 | 0.9014 | -14.53 | 0.6958 | 0.36 | 0.07494 | 0 | 0 | 0 | 0 | 0.2898 | 0.1808 |
| scalar | gap | drifting | kh_dual_control | 3 | -12.73 | 0.2019 | 12.73 | 0.2019 | -12.08 | 0.3059 | 0.432 | 0.02078 | 0 | 0 | 0 | 0 | 0.2244 | 0.1759 |
| scalar | gap | drifting | tv_gp_lcb | 3 | -2.385e+04 | 2779 | 2.385e+04 | 2779 | -2.204e+04 | 2747 | 28.84 | 1.102 | 0 | 0 | 0 | 0 | 1785 | 123.1 |
| scalar | gap | piecewise | arcari_dual_smpc | 3 | -23.64 | 6.871 | 23.64 | 6.871 | -23.01 | 6.936 | 0.444 | 0.09372 | 0 | 0 | 0 | 0 | 0.1887 | 0.1426 |
| scalar | gap | piecewise | kh_dual_control | 3 | -10.48 | 1.371 | 10.48 | 1.371 | -9.919 | 1.437 | 0.384 | 0.07869 | 0 | 0 | 0 | 0 | 0.1808 | 0.1464 |
| scalar | gap | piecewise | tv_gp_lcb | 3 | -6.866e+04 | 3.409e+04 | 6.866e+04 | 3.409e+04 | -6.521e+04 | 3.137e+04 | 29.88 | 2.054 | 0 | 0 | 0 | 0 | 3412 | 2803 |
| scalar | gap | static | arcari_dual_smpc | 3 | -22.93 | 5.272 | 22.93 | 5.272 | -22.54 | 5.39 | 0.348 | 0.146 | 0 | 0 | 0 | 0 | 0.04869 | 0.01209 |
| scalar | gap | static | kh_dual_control | 3 | -12.81 | 1.755 | 12.81 | 1.755 | -11.86 | 1.8 | 0.348 | 0.03175 | 0 | 0 | 0 | 0 | 0.593 | 0.1639 |
| scalar | gap | static | tv_gp_lcb | 3 | -7.659e+04 | 2.528e+04 | 7.659e+04 | 2.528e+04 | -7.299e+04 | 2.37e+04 | 30.65 | 1.435 | 0 | 0 | 0 | 0 | 3568 | 1624 |
| scalar | no_gap | drifting | arcari_dual_smpc | 3 | -14.98 | 1.74 | 14.98 | 1.74 | -14.53 | 1.734 | 0.324 | 0.1157 | 0 | 0 | 0 | 0 | 0.1239 | 0.1072 |
| scalar | no_gap | drifting | kh_dual_control | 3 | -13.34 | 1.947 | 13.34 | 1.947 | -12.82 | 1.955 | 0.396 | 0.09525 | 0 | 0 | 0 | 0 | 0.1218 | 0.1036 |
| scalar | no_gap | drifting | tv_gp_lcb | 3 | -7532 | 646.3 | 7532 | 646.3 | -7492 | 647.4 | 27.19 | 1.245 | 0 | 0 | 0 | 0 | 12.38 | 5.059 |
| scalar | no_gap | piecewise | arcari_dual_smpc | 3 | -24.75 | 6.947 | 24.75 | 6.947 | -24.13 | 6.944 | 0.372 | 0.07299 | 0 | 0 | 0 | 0 | 0.2452 | 0.1058 |
| scalar | no_gap | piecewise | kh_dual_control | 3 | -12.25 | 1.109 | 12.25 | 1.109 | -11.51 | 0.925 | 0.384 | 0.024 | 0 | 0 | 0 | 0 | 0.3647 | 0.2078 |
| scalar | no_gap | piecewise | tv_gp_lcb | 3 | -1.209e+04 | 5706 | 1.209e+04 | 5706 | -1.175e+04 | 5528 | 29.46 | 0.8013 | 0 | 0 | 0 | 0 | 306.4 | 184.5 |
| scalar | no_gap | static | arcari_dual_smpc | 3 | -22.17 | 5.529 | 22.17 | 5.529 | -21.71 | 5.665 | 0.252 | 0.07494 | 0 | 0 | 0 | 0 | 0.2051 | 0.1123 |
| scalar | no_gap | static | kh_dual_control | 3 | -9.887 | 0.6737 | 9.887 | 0.6737 | -9.406 | 0.6468 | 0.276 | 0.05231 | 0 | 0 | 0 | 0 | 0.2051 | 0.1123 |
| scalar | no_gap | static | tv_gp_lcb | 3 | -1.077e+04 | 2128 | 1.077e+04 | 2128 | -1.033e+04 | 2213 | 30.16 | 0.7276 | 0 | 0 | 0 | 0 | 413.1 | 85.73 |

## CartPole summary
| environment | twin_gap | regime | baseline | n | mean_net_reward | stderr_net_reward | mean_total_cost | stderr_total_cost | mean_acc_task_reward | stderr_acc_task_reward | mean_acc_energy_cost | stderr_acc_energy_cost | mean_acc_switch_cost | stderr_acc_switch_cost | mean_acc_failure_cost | stderr_acc_failure_cost | mean_terminal_cost | stderr_terminal_cost | mean_failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole | gap | drifting | arcari_dual_smpc | 2 | -0.9332 | 0.01372 | 0.9332 | 0.01372 | -0.7537 | 0.02079 | 0.019 | 0.001 | 0.15 | 0.01 | 0 | 0 | 0.0105 | 0.003927 | 0 |
| cartpole | gap | drifting | kh_dual_control | 2 | -11.51 | 0.1838 | 11.51 | 0.1838 | -8.403 | 0.156 | 0 | 0 | 0 | 0 | 0 | 0 | 3.105 | 0.02783 | 0 |
| cartpole | gap | drifting | tv_gp_lcb | 2 | -839.4 | 0.4213 | 839.4 | 0.4213 | -32.47 | 0.3537 | 0.022 | 0 | 0.46 | 0 | 800 | 0 | 6.465 | 0.06761 | 8 |
| cartpole | gap | piecewise | arcari_dual_smpc | 2 | -0.8535 | 0.02494 | 0.8535 | 0.02494 | -0.6826 | 0.02911 | 0.017 | 0.002 | 0.135 | 0.005 | 0 | 0 | 0.01889 | 0.002823 | 0 |
| cartpole | gap | piecewise | kh_dual_control | 2 | -12.6 | 2.089 | 12.6 | 2.089 | -9.22 | 1.421 | 0 | 0 | 0 | 0 | 0 | 0 | 3.377 | 0.6677 | 0 |
| cartpole | gap | piecewise | tv_gp_lcb | 2 | -536.6 | 101 | 536.6 | 101 | -28.79 | 1.206 | 0.0235 | 0.0005 | 0.515 | 0.035 | 500 | 100 | 7.283 | 0.1568 | 5 |
| cartpole | gap | static | arcari_dual_smpc | 2 | -0.8733 | 0.03486 | 0.8733 | 0.03486 | -0.7091 | 0.01498 | 0.0145 | 0.0025 | 0.125 | 0.005 | 0 | 0 | 0.02471 | 0.01237 | 0 |
| cartpole | gap | static | kh_dual_control | 2 | -12.6 | 2.089 | 12.6 | 2.089 | -9.22 | 1.421 | 0 | 0 | 0 | 0 | 0 | 0 | 3.377 | 0.6677 | 0 |
| cartpole | gap | static | tv_gp_lcb | 2 | -638.3 | 104.1 | 638.3 | 104.1 | -30.69 | 3.713 | 0.022 | 0 | 0.46 | 0.01 | 600 | 100 | 7.139 | 0.3597 | 6 |
| cartpole | no_gap | drifting | arcari_dual_smpc | 2 | -0.6544 | 0.01547 | 0.6544 | 0.01547 | -0.5426 | 0.01867 | 0.0085 | 0.0005 | 0.095 | 0.005 | 0 | 0 | 0.008274 | 0.002293 | 0 |
| cartpole | no_gap | drifting | kh_dual_control | 2 | -11.5 | 0.1849 | 11.5 | 0.1849 | -8.4 | 0.1568 | 0 | 0 | 0 | 0 | 0 | 0 | 3.102 | 0.02806 | 0 |
| cartpole | no_gap | drifting | tv_gp_lcb | 2 | -791.7 | 48.45 | 791.7 | 48.45 | -34.2 | 0.9633 | 0.0225 | 0.0005 | 0.49 | 0 | 750 | 50 | 6.961 | 0.5883 | 7.5 |
| cartpole | no_gap | piecewise | arcari_dual_smpc | 2 | -0.6232 | 0.03784 | 0.6232 | 0.03784 | -0.5103 | 0.04459 | 0.0065 | 0.0025 | 0.07 | 0.02 | 0 | 0 | 0.03636 | 0.02926 | 0 |
| cartpole | no_gap | piecewise | kh_dual_control | 2 | -12.6 | 2.09 | 12.6 | 2.09 | -9.221 | 1.422 | 0 | 0 | 0 | 0 | 0 | 0 | 3.378 | 0.6678 | 0 |
| cartpole | no_gap | piecewise | tv_gp_lcb | 2 | -689.7 | 51.18 | 689.7 | 51.18 | -32.1 | 1.272 | 0.022 | 0 | 0.45 | 0 | 650 | 50 | 7.084 | 0.09524 | 6.5 |
| cartpole | no_gap | static | arcari_dual_smpc | 2 | -0.6208 | 0.03263 | 0.6208 | 0.03263 | -0.5141 | 0.0007779 | 0.008 | 0.003 | 0.085 | 0.025 | 0 | 0 | 0.01375 | 0.00541 | 0 |
| cartpole | no_gap | static | kh_dual_control | 2 | -12.6 | 2.09 | 12.6 | 2.09 | -9.221 | 1.422 | 0 | 0 | 0 | 0 | 0 | 0 | 3.378 | 0.6678 | 0 |
| cartpole | no_gap | static | tv_gp_lcb | 2 | -793 | 50.6 | 793 | 50.6 | -35.3 | 0.8205 | 0.0225 | 0.0005 | 0.49 | 0 | 750 | 50 | 7.167 | 0.2232 | 7.5 |