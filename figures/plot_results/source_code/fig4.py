import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import source_code.ptitprince1 as pt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, mannwhitneyu
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'


def stats_test(df, tertile_var, probabilities):
    # noramlity test
    for tertile in df[tertile_var].unique():
        data = df[df[tertile_var] == tertile][probabilities]
        stat, p = stats.shapiro(data)
        print(f"Normality test for {tertile}: Statistics={stat:.2f}, p={p:.2e}")

    # homogeneity of variances
    grouped_data = [df[df[tertile_var] == tertile][probabilities] for tertile in df[tertile_var].unique()]
    stat, p = stats.levene(*grouped_data)
    print(f"Levene's test for homogeneity of variances: Statistics={stat:.2f}, p={p:.2e}")

def mann_whitney_test_tau(df, group_col, value_col):
    groups = df[group_col].unique()
    
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups for Mann-Whitney test, but found {len(groups)}")
    
    group1_data = df[df[group_col] == groups[0]][value_col]
    group2_data = df[df[group_col] == groups[1]][value_col]
    
    statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='less')
    
    if value_col == 'amy_CENTILOIDS':
        table_value = 'CL' 
    elif value_col == 'amy_label_prob':
        table_value = 'P(Aβ)' 
    elif value_col == 'tau_META_VILLE_SUVR':
        table_value = 'Meta-τ SUVR'
    elif value_col == 'tau_label_prob':
        table_value = 'P(τ)' 
    result_df = pd.DataFrame({
        'Measure': [table_value],
        'U Statistic': [statistic],
        'p-value': [f'{p_value:.2e}']
    })

    return result_df


def add_stat_significance(fig, x_start, x_end, y_start, y_end, text, orientation):
    if orientation == 'horizontal':
        # Horizontal Line
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_end, y1=y_start, line=dict(color="black", width=2))
        # Ticks
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_start, y1=y_start + 0.02, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=x_end, y0=y_start, x1=x_end, y1=y_start + 0.02, line=dict(color="black", width=2))
        # Text
        fig.add_annotation(x=(x_start + x_end) / 2, y=y_start + 0.02, text=text, showarrow=False, font=dict(size=14))
    else:
        # Vertical Line
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_start, y1=y_end, line=dict(color="black", width=2))
        # Ticks
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_start - 0.02, y1=y_start, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=x_start, y0=y_end, x1=x_start - 0.02, y1=y_end, line=dict(color="black", width=2))
        # Text
        fig.add_annotation(x=x_start - 0.03, y=(y_start + y_end) / 2, text=text, showarrow=False, font=dict(size=14), textangle=-90)


def rainclouds_tau_levels(df, figname):
    sns.set_theme(style="white", context="paper")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 2.3), sharey=True)
    font_sizes = 7

    # centiloids
    cb_palette1 = sns.color_palette("inferno_r")
    custom_pal1 = {'Low/med τ PET': cb_palette1[0], 'High τ PET': cb_palette1[2]}

    pt.RainCloud(data = df, x = "tau_level", y = "amy_CENTILOIDS", orient='h', 
    palette = custom_pal1, bw=.2, ax=ax1, move = .2, linewidth=0.5, dodge=True, 
    width_viol=.7, width_box=0.2, point_size = 1, jitter=1, cut=3)

    ax1.plot([300, 300], [0, 1], color='black', linewidth=2) # vert line
    ax1.plot([300, 294], [0, 0], color='black', linewidth=2) # tick 1
    ax1.plot([300, 294], [1, 1], color='black', linewidth=2) # tick 2
    ax1.text(308, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars

    if ax1.get_yticks().size > 0:
        positions = ax1.get_yticks()
        labels = ['Low/med τ PET', 'High τ PET']
        ax1.set_yticks(positions[:len(labels)])
        ax1.set_yticklabels(labels)

    ax1.set_xlabel("Centiloids", fontname='Arial', fontsize=font_sizes)
    ax1.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax1.tick_params(axis='both', labelsize=font_sizes)

    for c in ax1.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    # prob amy
    cb_palette = sns.color_palette("YlGnBu")
    custom_pal = {'Low/med τ PET': cb_palette[0], 'High τ PET': cb_palette[2]}

    pt.RainCloud(data = df, x = "tau_level", y = "amy_label_prob", orient='h', cut=3,
    palette = custom_pal, bw=.2, ax=ax2, move = .2, linewidth=0.5, dodge=True,
    width_viol=.7, width_box=0.2, point_size = 1, jitter=1)

    ax2.plot([0.81, 0.81], [0, 1], color='black', linewidth=2) # vert line
    ax2.plot([0.81, 0.80], [0, 0], color='black', linewidth=2) # tick 1
    ax2.plot([0.81, 0.80], [1, 1], color='black', linewidth=2) # tick 2
    ax2.text(0.83, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars

    ax2.set_xlabel("P(Aβ)", fontname='Arial', fontsize=font_sizes)
    ax2.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax2.tick_params(axis='both', labelsize=font_sizes)

    # makes lines thinner
    for c in ax2.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    sns.despine()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()


def rainclouds_cl_levels(df, figname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 2.3), sharey=True)
    font_sizes = 7

    # centiloids
    pt.RainCloud(data = df, x = "cl_level", y = "tau_META_VILLE_SUVR", orient='h', 
    palette = "BuPu", bw=.2, ax=ax1, move = .2, linewidth=0.5, dodge=True, 
    width_viol=.7, width_box=0.2, point_size = 1, jitter=1, cut=3)

    ax1.plot([3, 3], [0, 1], color='black', linewidth=2) # vert line
    ax1.plot([3, 2.97], [0, 0], color='black', linewidth=2) # tick 1
    ax1.plot([3, 2.97], [1, 1], color='black', linewidth=2) # tick 2
    ax1.text(3.05, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars

    ax1.set_xlabel("Meta-τ SUVR", fontname='Arial', fontsize=font_sizes)
    ax1.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax1.tick_params(axis='both', labelsize=font_sizes)

    for c in ax1.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    # probs
    cb_palette = sns.color_palette("magma_r")
    custom_pal = {'Low/med CL': cb_palette[3], 'High CL': cb_palette[5]}

    pt.RainCloud(data=df, x="cl_level", y="tau_label_prob", orient='h', cut=3,
                palette=cb_palette, bw=.2, ax=ax2, move=.2, linewidth=0.5, dodge=True,
                width_viol=.7, width_box=0.2, point_size=1, jitter=1)

    ax2.plot([1, 1], [0, 1], color='black', linewidth=2) # vert line
    ax2.plot([1, 0.98], [0, 0], color='black', linewidth=2) # tick 1
    ax2.plot([1, 0.98], [1, 1], color='black', linewidth=2) # tick 2
    ax2.text(1.025, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars


    ax2.set_xlabel("P(τ)", fontname='Arial', fontsize=font_sizes)
    ax2.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax2.tick_params(axis='both', labelsize=font_sizes)

    # makes lines thinner
    for c in ax2.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    sns.despine()
    plt.tight_layout()

    plt.savefig(figname, dpi=300)
    plt.show()


def kde_plot(df, figname):
    pio.kaleido.scope.mathjax = None
    font_sizes = 9

    cohort_markers = {'ADNI': 'circle', 'HABS': 'cross', 'NACC': 'diamond'}
    color_map = {'Aβ+, τ+': '#CC503E', 'Aβ-, τ-': '#008080'}

    # making subplots to have more control on spacing between boxplots: 2x2 grid, top-right empty
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.95, 0.15],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[[{"type": "box"}, None],
               [{"type": "contour"}, {"type": "box"}]]
    )

    # main contour plot (bottom left)
    contour_fig = px.density_contour(
        df,
        x='amy_label_prob',
        y='tau_label_prob',
        color='Profile',
        marginal_x=None,
        marginal_y=None,
        color_discrete_map=color_map,
        title=''
    )
    for trace in contour_fig.data:
        fig.add_trace(trace, row=2, col=1)


    # top marginal box plot (x)
    for profile, color in color_map.items():
        fig.add_trace(
            go.Box(
                x=df.loc[df['Profile'] == profile, 'amy_label_prob'],
                marker_color=color,
                name=profile,
                boxpoints='outliers',
                marker_size=2,
                line_width=0.7,
                showlegend=False,
                width=0.4
            ),
            row=1, col=1
        )

    # right marginal box plot (y)
    for profile, color in color_map.items():
        fig.add_trace(
        go.Box(
            y=df.loc[df['Profile'] == profile, 'tau_label_prob'],
            marker_color=color,
            name=profile,
            boxpoints='outliers',
            marker_size=2,
            line_width=0.7,
            showlegend=False,
            width=0.2
        ),
        row=2, col=2
    )

    # custom traces (contour and box)
    for trace in fig.data:
        if trace.type == 'histogram2dcontour':
            trace.line.width = 0.7
            trace.showlegend = True
        elif trace.type == 'box':
            trace.showlegend = False
            trace.line.width = 0.7
            trace.boxpoints = 'outliers'
            trace.pointpos = 0
            trace.jitter = 0.3
            trace.marker.size = 2
            trace.notched = False

    # add scatter
    fig.add_trace(go.Scatter(
        x=df['amy_label_prob'],
        y=df['tau_label_prob'],
        mode='markers',
        marker=dict(
            size=2.5,
            color=df['Profile'].map(color_map),
            symbol=df['COHORT'].map(cohort_markers)
        ),
        showlegend=False
    ), row=2, col=1)

    # legend for cohort markers
    for cohort, marker in cohort_markers.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=2.5,
                color='black',
                symbol=marker
            ),
            name=cohort,
            showlegend=True
        ))

    add_manual_significance_lines(fig)

    # Layout and axes
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=10, b=10, pad=0),
        legend=dict(
            title_text="",
            yanchor="top",
            y=1.25,
            xanchor="center",
            x=0.5,
            orientation="h",
            itemsizing="trace",
            font=dict(size=font_sizes, family='Arial', color='black')
        ),
        font=dict(size=font_sizes, family='Arial', color='black')
    )

    fig.update_xaxes(title_text='P(Aβ)', row=2, col=1, showline=True, linecolor='black', linewidth=0.7, nticks=10, ticklen=2, title_font=dict(family="Arial", size=font_sizes))
    fig.update_yaxes(title_text='P(τ)', row=2, col=1, showline=True, linecolor='black', linewidth=0.7, nticks=10, ticklen=2, title_font=dict(family="Arial", size=font_sizes))
    fig.update_xaxes(showticklabels=False, title_text='', row=2, col=2)
    fig.update_xaxes(showticklabels=False, title_text='', row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text='', row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text='', row=2, col=2)

    pio.write_image(fig, figname, width=442, height=221)


def add_manual_significance_lines(fig):
    """Manually add significance lines with 4 stars (****)"""
    
    # x axis
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0.88, y0=0.82,  
        x1=0.975, y1=0.82,  
        line=dict(color="black", width=0.7)
    )
    
    # caps for x line
    fig.add_shape(type="line", xref="paper", yref="paper",
                x0=0.881, y0=0.82, x1=0.881, y1=0.80,  # left
                line=dict(color="black", width=0.7))
    fig.add_shape(type="line", xref="paper", yref="paper",
                x0=0.974, y0=0.82, x1=0.974, y1=0.80,  # right cap
                line=dict(color="black", width=0.7))
    
    # stars
    fig.add_annotation(xref="paper", yref="paper", x=0.95, y=0.89,
                    text="****", showarrow=False,
                    font=dict(size=9, color="black"))
    
    # y axis
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0.83, y0=0.96,  
        x1=0.83, y1=0.83, 
        line=dict(color="black", width=0.7)
    )
    
    # end caps for y
    fig.add_shape(type="line", xref="paper", yref="paper",
                x0=0.831, y0=0.83, x1=0.823, y1=0.83, # bottom cap
                line=dict(color="black", width=0.7))

    fig.add_shape(type="line", xref="paper", yref="paper",
                x0=0.831, y0=0.96, x1=0.823, y1=0.96,  # top cap
                line=dict(color="black", width=0.7))

    # stars
    fig.add_annotation(xref="paper", yref="paper", x=0.855, y=0.96,
                    text="****", showarrow=False, textangle=90,
                    font=dict(size=9, color="black"))
               

def plot(config):
    # Figure 4a
    tau_level_df = pd.read_csv(config['source_data']['fig4a'])
    rainclouds_tau_levels(tau_level_df, figname=config['output']['fig4a'])
    # Figure 4b
    cl_level_df = pd.read_csv(config['source_data']['fig4b'])
    rainclouds_cl_levels(cl_level_df, figname=config['output']['fig4b'])
    # Figure 4c
    df = pd.read_csv(config['source_data']['fig4c'])           
    kde_plot(df, figname=config['output']['fig4c'])