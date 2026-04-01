# Headline: Stop Guessing, Start Understanding: A Smarter Way to Read the Stock Market<p><p>

## Hook

Every single day, the stock market generates a massive amount of information in the form of prices, trading volumes, news headlines, 
earnings reports. Investors, analysts, and algorithms all race to make sense of it. But the harsh reality is that many financial analysis
tools are still insufficient to accurately predict what is going to happen tomorrow. The question
isn't whether we can eliminate that uncertainty. It's whether we can harness it and inform our decisions around it.

## Problem Statement

Traditional stock prediction tools give you an exact number: "This stock will be $187.42 tomorrow." That sounds precise and confident, 
but that confidence is an illusion. Stock prices are moved by an enormous tangle of forces such as company earnings, breaking news, 
broader economic shifts, and sometimes just the collective mood of millions of traders making decisions at once. A model which gives 
certain and exact answers isn't being precise, it is just forced to choose one possibility in a wide range of uncertain outcomes.
Without knowing how uncertain this exact prediction is, financial analysts cannot make informed decisions.

## Solution Description

My solution proposes throwing away single value predictions and instead training a model to understand how much the value could fluctuate in the future.
In other words, it can tell you, with a degree of certainty, where it thinks the stock price will move in the next day, the following, or the next week.
In addition to this, news articles on current events regarding each publicly traded company is analyzed to understand if there is a positive or negative
attitude about that company. This may truly impact the activity of the company, or just how people perceive them, and both are important to consider. 
By incorporating both numerical and non-numerical data, and including consideration for uncertainty, we can make well-informed decision with the model.

## Visualization

This graph is an output of the model showing how it measures uncertainty in relation to the price of a stock. 
One important thing to note is the cone of uncertainty follows a likely trajectory, showing the model isn't just guessing.
This is the specific visualization for the stock JNJ, or Johnson & Johnson, displaying the Price History, with the forecasted path,
overlayed by useful technical indicators, and a chart of the volume of trades at the bottom. The top right displays the suggested action.

<img src="plots/ticker_JNJ.png" width="75%"/>
