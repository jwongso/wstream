#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "audio_source.h"

// Mock implementation of audio_source for testing
class mock_audio_source : public audio_source {
public:
    MOCK_METHOD(bool, initialize, (), (override));
    MOCK_METHOD(bool, start, (), (override));
    MOCK_METHOD(void, stop, (), (override));
    MOCK_METHOD(bool, get_audio_samples, (std::vector<float>&), (override));
    MOCK_METHOD(std::string, get_name, (), (const, override));
    MOCK_METHOD(bool, is_active, (), (const, override));
    MOCK_METHOD(std::string, get_session_id, (), (const, override));
    MOCK_METHOD(std::string, get_language, (), (const, override));
};

TEST(AudioSourceTest, DefaultSessionIdIsEmpty) {
    mock_audio_source source;
    EXPECT_EQ(source.get_session_id(), "");
}

TEST(AudioSourceTest, DefaultLanguageIsEmpty) {
    mock_audio_source source;
    EXPECT_EQ(source.get_language(), "");
}

TEST(AudioSourceTest, MockMethodsWork) {
    mock_audio_source source;

    // Set up expectations
    EXPECT_CALL(source, initialize()).WillOnce(testing::Return(true));
    EXPECT_CALL(source, start()).WillOnce(testing::Return(true));
    EXPECT_CALL(source, stop());
    EXPECT_CALL(source, get_audio_samples(testing::_)).WillOnce(testing::Return(false));
    EXPECT_CALL(source, get_name()).WillOnce(testing::Return("mock"));
    EXPECT_CALL(source, is_active()).WillOnce(testing::Return(false));

    // Call methods
    EXPECT_TRUE(source.initialize());
    EXPECT_TRUE(source.start());
    source.stop();
    std::vector<float> samples;
    EXPECT_FALSE(source.get_audio_samples(samples));
    EXPECT_EQ(source.get_name(), "mock");
    EXPECT_FALSE(source.is_active());
}
