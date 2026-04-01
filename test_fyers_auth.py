from fyers_apiv3 import fyersModel
import credentials as cd


def main() -> None:
    with open("access.txt", "r", encoding="utf-8") as f:
        access_token = f.read().strip()

    print("client_id:", cd.client_id)
    print("token_prefix:", access_token[:24] + "...")

    fyers = fyersModel.FyersModel(
        client_id=cd.client_id,
        is_async=False,
        token=access_token,
        log_path="",
    )

    print("\nFUNDS RESPONSE:")
    try:
        funds = fyers.funds()
        print(funds)
    except Exception as exc:
        print("FUNDS error:", exc)

    print("\nQUOTES RESPONSE (NSE:RELIANCE):")
    try:
        quotes = fyers.quotes({"symbols": "NSE:RELIANCE"})
        print(quotes)
    except Exception as exc:
        print("QUOTES error:", exc)

    print("\nQUOTES RESPONSE (NSE:RELIANCE-EQ):")
    try:
        quotes_eq = fyers.quotes({"symbols": "NSE:RELIANCE-EQ"})
        print(quotes_eq)
        try:
            d = (quotes_eq or {}).get("d") or []
            if d:
                v = d[0].get("v") or {}
                print("Parsed LTP: lp=", v.get("lp"), "ltp=", v.get("ltp"))
        except Exception:
            pass
    except Exception as exc:
        print("QUOTES -EQ error:", exc)


if __name__ == "__main__":
    main()
