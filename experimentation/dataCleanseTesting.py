def price_to_float(price_str):
    if "-" in price_str:
        lower_limit = price_str[1: price_str.index('-') - 1]
        lower_limit = float(lower_limit.replace(',', ''))
        upper_limit = price_str[price_str.index('-') + 3:]
        upper_limit = float(upper_limit.replace(',', ''))
        price_float = (upper_limit + lower_limit) / 2
    elif "$" in price_str:
        price_str = price_str.replace(',', '')
        price_float = float(price_str.replace('$', ''))
    else:
        price_float = 0
    return price_float


price_to_float("$5.50")
price_to_float("$5.50 - $6.99")
price_to_float("{} xjkjxkjkjx")
price_to_float()