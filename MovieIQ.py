from pathlib import Path
import pandas as pd, streamlit as st, joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

st.set_page_config(page_title="MovieIQ",page_icon="🎬",layout="wide")
BASE=Path(__file__).resolve().parent
@st.cache_data
def load_data():
    d=pd.read_csv(BASE/"movies.csv")
    d["genre_list"]=d["genres"].fillna("Unknown").apply(lambda x:[g for g in str(x).split("|") if g] or ["Unknown"])
    d["success"]=(d.revenue>d.budget).astype(int)
    return d
@st.cache_resource
def load_model(): return joblib.load(BASE/"models"/"movieiq_random_forest.joblib")
df=load_data(); model=load_model()

st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption("Success rule: revenue > budget")
genres=sorted({g for row in df.genre_list for g in row})
genre=st.sidebar.selectbox("Genre",["All"]+genres)
min_vote=st.sidebar.slider("Minimum vote average",0.0,10.0,0.0,.1)
f=df[df.vote_average>=min_vote].copy()
if genre!="All": f=f[f.genre_list.apply(lambda x:genre in x)]

a,b,c,d=st.columns(4)
a.metric("Movies",f"{len(f):,}")
b.metric("Success rate",f"{f.success.mean()*100:.1f}%" if len(f) else "N/A")
c.metric("Median budget",f"${f.budget.median()/1e6:.1f}M" if len(f) else "N/A")
d.metric("Median revenue",f"${f.revenue.median()/1e6:.1f}M" if len(f) else "N/A")

t1,t2,t3,t4=st.tabs(["Data","EDA","Tests","Prediction"])
with t1:
    st.dataframe(f[["title","primary_genre","budget","revenue","popularity","runtime","vote_average","success"]],use_container_width=True,hide_index=True)
with t2:
    if f.empty: st.warning("No matching movies.")
    else:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots();sns.scatterplot(data=f,x="budget",y="revenue",hue="success",ax=ax,alpha=.7)
            ax.set_xscale("log");ax.set_yscale("log");ax.set_title("Budget vs Revenue");st.pyplot(fig)
        with c2:
            ex=f.assign(genre=f.genre_list).explode("genre")
            s=ex.genre.value_counts().head(10).sort_values()
            fig,ax=plt.subplots();ax.barh(s.index,s.values);ax.set_title("Most Common Genres");st.pyplot(fig)
        st.image(str(BASE/"assets"/"correlation_heatmap.png"))
        st.image(str(BASE/"assets"/"feature_importance.png"))
with t3:
    x=df.loc[df.success==1,"popularity"];y=df.loc[df.success==0,"popularity"]
    ts,tp=ttest_ind(x,y,equal_var=False)
    ex=df.assign(genre=df.genre_list).explode("genre");ch,cp,dof,_=chi2_contingency(pd.crosstab(ex.genre,ex.success))
    st.write(f"Welch T-Test p-value: **{tp:.6g}**")
    st.write("Popularity differs significantly." if tp<.05 else "No significant popularity difference.")
    st.write(f"Chi-Square p-value: **{cp:.6g}**")
    st.write("Genre and success are associated." if cp<.05 else "No significant genre association.")
with t4:
    with st.form("predict"):
        budget=st.number_input("Budget (USD)",min_value=1000.0,value=30000000.0,step=1000000.0)
        popularity=st.number_input("Popularity",min_value=0.0,value=25.0)
        runtime=st.number_input("Runtime (minutes)",min_value=1.0,value=110.0)
        vote=st.slider("Vote average",0.0,10.0,6.5,.1)
        pg=st.selectbox("Primary genre",sorted(df.primary_genre.unique()))
        go=st.form_submit_button("Predict")
    if go:
        row=pd.DataFrame([{"budget":budget,"popularity":popularity,"runtime":runtime,"vote_average":vote,"primary_genre":pg}])
        p=model.predict_proba(row)[0,1]
        (st.success if p>=.5 else st.error)(f"{'Likely successful' if p>=.5 else 'Likely not successful'} — success probability {p:.1%}")
        st.caption("Educational estimate, not a financial guarantee.")
