def cal_market_capital(price, shares_outstanding):
    return price * shares_outstanding

if __name__ == "__main__":
    price = 1245.0 # stock price
    shares_outstanding = 23634
    company_market_capt = cal_market_capital(price, shares_outstanding)
    print(f"The market capitalizationof the company is : ${company_market_capt}")
