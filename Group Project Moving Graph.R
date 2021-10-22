#Installing Required Packages
library(ggplot2)
library(tidyverse)
library(gganimate)
library(gifski)
library(sqldf)
library(plyr)
library(magrittr)
library(dplyr)

#Setting Working Directory
setwd("C://Users/Alec/Desktop/Syracuse Data Analytics/Applied Data Science/Group Project")

df_HOF_Career <- read.csv("C://Users/Alec/Desktop/Syracuse Data Analytics/Applied Data Science/Group Project/df_HOF_career")

df_HOF_Career$HOF_probability <- predict(cs_model, df_HOF_Career, type="response", na.rm=TRUE)
df_HOF_Career$HOF_probability_PG <- predict(pos_pg_model, df_HOF_Career, type="response", na.rm=TRUE)
df_HOF_Career$HOF_probability_SG <- predict(pos_sg_model, df_HOF_Career, type="response", na.rm=TRUE)
df_HOF_Career$HOF_probability_F <- predict(pos_any_f_model, df_HOF_Career, type="response", na.rm=TRUE)
df_HOF_Career$HOF_probability_C <- predict(pos_c_model, df_HOF_Career, type="response", na.rm=TRUE)

write.csv("C://Users/Alec/Desktop/Syracuse Data Analytics/Applied Data Science/Group Project/df_HOF_Career_prob.csv")

#THE PLOT
df_HOF_Career_Probs <- read.csv("C://Users/Alec/Desktop/Syracuse Data Analytics/Applied Data Science/Group Project/df_HOF_Career_prob.csv")

THE_PLOT <- ggplot(df_HOF_Career_Probs, aes(HOF_probability, POS_Prob, size = PTS, color = Pos)) + 
            geom_point() + scale_size(range = c(1, 10)) +
            geom_text(aes(label = df_HOF_Career_Probs$Player)) + xlim(0, 1.25) +
            labs(title = 'Year: {frame_time}', x = 'HOF_probability', y = "POS_Prob") +
            transition_time(Year) + ease_aes('linear') + xlab("HOF Probability") + ylab("Player Position HOF Probability")

animate(THE_PLOT, duration = 15, fps = 5, width = 500, height = 500, renderer = gifski_renderer())
anim_save("nba_output.gif")