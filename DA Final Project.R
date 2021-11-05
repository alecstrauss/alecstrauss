#Load in Packages
library("arules")
library("arulesViz")
library("rpart")
library("rpart.plot")

"Association Rule Mining"
  #Load in the data
movie_genres <- read.transactions("C://Users/Alec/Desktop/Syracuse Data Analytics/Data Analytics/Final Project/Movie Genres.csv", sep = ",")
genres_revenue <- read.transactions("C://Users/Alec/Desktop/Syracuse Data Analytics/Data Analytics/Final Project/Movie Genres and Revenue.csv", sep = ",")

  #Explore the data
itemFrequencyPlot(movie_genres, topN = 20, type = "absolute")

  #Generate rules
genre_rules <- apriori(movie_genres, parameter = list(supp = 0.01, conf = 0.8))
genre_rules

inspect(genre_rules)

revenue_rules <- apriori(genres_revenue, parameter = list(supp = 0.01, conf = 0.8))
revenue_rules

inspect(revenue_rules)

HR_rule <- apriori(data = genres_revenue, parameter = list(supp = 0.0001, conf = 0.8), appearance = list(default = "lhs", 
                   rhs = "Highest Revenue"), control = list(verbose = F))
inspect(HR_rule[1:5])

HR_rule_support <- sort(HR_rule, decreasing = TRUE, by = "support")
inspect(HR_rule_support[1:5])

HR_rule_confidence <- sort(HR_rule, decreasing = TRUE, by = "confidence")
inspect(HR_rule_confidence[1:5])

HR_rule_lift <- sort(HR_rule, decreasing = TRUE, by = "lift")
inspect(HR_rule_confidence[1:5])




new_rules <- apriori(data = movie_genres, parameter = list(supp = 0.001, conf = 0.08), appearance = list(default = "lhs",
                      rhs = "Fantasy"), control = list(verbose = F))


"Decision Tree"
#Load in the data
movie_dt <- read.csv("C://Users/Alec/Desktop/Syracuse Data Analytics/Data Analytics/Final Project/movie decision tree.csv")

#Change the data to factors
movie_dt$revenue <- as.factor(movie_dt$revenue)
movie_dt$budget <- as.factor(movie_dt$budget)
movie_dt$original_language <- as.factor(movie_dt$original_language)
movie_dt$release_month <- as.factor(movie_dt$release_month)
movie_dt$runtime_dummy <- as.factor(movie_dt$runtime_dummy)

str(movie_dt)

#Create a Training and Testing Data Set
create_train_test <- function(data, size = 0.8, train = TRUE){
  n_row = nrow(data)
  total_row = size * n_row
  train_sample <- 1: total_row
  if(train == TRUE){
    return(data[train_sample,])
  } else{
    return(data[-train_sample,])
  }
}

dt_train <- create_train_test(movie_dt, 0.8, train = TRUE)
dt_test <- create_train_test(movie_dt, 0.8, train = FALSE)

#Building the Decision Tree Model
dt_plot <- rpart(revenue ~ budget, data = movie_dt, method = 'class')
rpart.plot(dt_plot, extra = 6)

dt_plot <- rpart(revenue ~ original_language, data = movie_dt, method = 'class')
rpart.plot(dt_plot, extra = 6)

dt_plot <- rpart(revenue ~ release_month, data = movie_dt, method = 'class')
rpart.plot(dt_plot, extra = 6)

dt_plot <- rpart(revenue ~ runtime_dummy, data = movie_dt, method = 'class')
rpart.plot(dt_plot, extra = 6)