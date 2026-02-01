import streamlit as st

# 1. Header
st.header("Online Feedback Portal")

# 2. Radio button
experience = st.radio(
    "How was your experience?",
    ["Excellent", "Good", "Average", "Poor"]
)

# 3. Slider
rating = st.slider("Rate us (1 to 10)", 1, 10)

# 4. Text area
feedback = st.text_area("Write your feedback")

# 5. Selectbox
recommend = st.selectbox(
    "Would you recommend us?",
    ["Yes", "No", "Maybe"]
)

# 6. Button
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
    st.write("### Feedback Summary")
    st.write(f"Experience: {experience}")
    st.write(f"Rating: {rating}/10")
    st.write(f"Recommendation: {recommend}")
    st.write(f"Comments: {feedback}")
