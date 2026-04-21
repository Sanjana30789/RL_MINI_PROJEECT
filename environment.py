import numpy as np


class LearningEnvironment:
    """
    Simulated student learning environment for RL-based recommendation.
    State: student knowledge vector across 5 topics (0 to 4)
    Action: index of content item to recommend
    Reward: based on match between content difficulty and student level
    """

    TOPICS = ["Math", "Science", "English", "History", "Coding"]
    N_TOPICS = 5
    N_LEVELS = 5
    N_CONTENT = 20
    MAX_STEPS = 50

    def __init__(self):
        self.content_library = []
        for topic in range(self.N_TOPICS):
            for diff in range(4):
                self.content_library.append(
                    {
                        "id": len(self.content_library),
                        "topic": topic,
                        "topic_name": self.TOPICS[topic],
                        "difficulty": diff,
                        "name": f"{self.TOPICS[topic]} Level {diff + 1}",
                    }
                )
        self.state = None
        self.step_count = 0
        self.reset()

    def reset(self):
        self.state = np.random.randint(0, 3, size=self.N_TOPICS)
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        content = self.content_library[action]
        topic = content["topic"]
        difficulty = content["difficulty"]
        student_level = int(self.state[topic])

        level_diff = difficulty - student_level

        if level_diff == 0:
            reward = 10.0
            if np.random.random() < 0.6:
                self.state[topic] = min(self.state[topic] + 1, self.N_LEVELS - 1)
        elif level_diff == 1:
            reward = 5.0
            if np.random.random() < 0.3:
                self.state[topic] = min(self.state[topic] + 1, self.N_LEVELS - 1)
        elif level_diff > 1:
            reward = -5.0
        elif level_diff == -1:
            reward = 2.0
        else:
            reward = -2.0

        if self.state[topic] == self.N_LEVELS - 1:
            reward += 15.0

        self.step_count += 1
        done = self.step_count >= self.MAX_STEPS

        info = {
            "topic": content["topic_name"],
            "topic_name": content["topic_name"],
            "difficulty": difficulty,
            "student_level": student_level,
            "content_name": content["name"],
            "name": content["name"],
        }

        return self.state.copy(), reward, done, info

    def state_to_index(self, state):
        index = 0
        for i, s in enumerate(state):
            index += int(s) * (self.N_LEVELS ** i)
        return index

    @property
    def action_size(self):
        return self.N_CONTENT

    @property
    def n_states(self):
        return self.N_LEVELS ** self.N_TOPICS