import numpy as np
import pandas as pd
import streamlit as st

from content_service import LESSON_BANK, QUIZ_BANK
from environment import LearningEnvironment
from train import train_q_learning


st.set_page_config(page_title="RL Learning Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #666;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1rem;
        border-radius: 14px;
        background: #f7f9fc;
        border: 1px solid #e6eaf2;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎓 RL Personalized Learning Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Quiz-based assessment → RL recommendation → lesson delivery</div>',
    unsafe_allow_html=True,
)

@st.cache_resource
def load_agent():
    agent, rewards = train_q_learning(500)
    return agent, rewards

agent, rewards = load_agent()
env = LearningEnvironment()

if "submitted" not in st.session_state:
    st.session_state.submitted = False

left, right = st.columns([1.1, 1])

with left:
    st.markdown("## 📝 Student Quiz")
    st.write("Answer these simple questions. The system will estimate topic-wise levels.")

    topic_scores = {}

    for topic in env.TOPICS:
        st.markdown(f"### {topic}")
        correct = 0
        questions = QUIZ_BANK[topic]

        for i, q in enumerate(questions):
            choice = st.radio(
                q["question"],
                q["options"],
                key=f"{topic}_{i}",
            )
            if choice == q["answer"]:
                correct += 1

        topic_scores[topic] = correct / len(questions)

    if st.button("🎯 Get Recommendation", use_container_width=True):
        st.session_state.submitted = True

        def score_to_level(score):
            if score == 1.0:
                return 4
            if score >= 0.75:
                return 3
            if score >= 0.5:
                return 2
            if score >= 0.25:
                return 1
            return 0

        levels = np.array([score_to_level(topic_scores[t]) for t in env.TOPICS])
        st.session_state.levels = levels

        env.state = levels.copy()
        state_idx = env.state_to_index(levels)
        action = agent.best_recommendation(state_idx)
        next_state, reward, _, info = env.step(action)

        st.session_state.recommendation = info
        st.session_state.reward = reward
        st.session_state.next_state = next_state

with right:
    st.markdown("## 📊 Student Dashboard")

    if st.session_state.submitted:
        levels = st.session_state.levels
        info = st.session_state.recommendation
        reward = st.session_state.reward
        next_state = st.session_state.next_state

        level_df = pd.DataFrame(
            {
                "Topic": env.TOPICS,
                "Estimated Level": levels,
            }
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Your Estimated Knowledge Levels")
        st.dataframe(level_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        topic = info["topic_name"]
        difficulty = info["difficulty"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📘 Recommended Lesson")
        st.success(f"{info['content_name']}")
        st.write(f"**Topic:** {topic}")
        st.write(f"**Difficulty:** Level {difficulty + 1}")
        st.write(f"**Reward:** {reward:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

        lesson = LESSON_BANK[topic][difficulty]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 Lesson Content")
        st.write(f"**{lesson['title']}**")
        st.write(lesson["content"])
        st.write(f"**Practice:** {lesson['practice']}")
        st.markdown("</div>", unsafe_allow_html=True)

        updated_df = pd.DataFrame(
            {
                "Topic": env.TOPICS,
                "Updated Level": next_state,
            }
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Updated Knowledge After Learning")
        st.dataframe(updated_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Complete the quiz to see personalized recommendation and lesson content.")