"""Synthetic corpus of POS help articles, menus and FAQs for a fictional product.

The corpus is the single source of truth: ``souschef.corpus`` renders it to
markdown files (the ingestion input) and derives the question/answer gold set
from the ``questions`` and ``keywords`` attached to every section.

All names, prices and policies are invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SectionSpec:
    heading: str
    body: str
    questions: tuple[str, ...]
    keywords: tuple[str, ...]  # facts a correct answer should contain


@dataclass(frozen=True)
class DocSpec:
    doc_id: str
    title: str
    category: str  # pos-help | menu | faq
    sections: tuple[SectionSpec, ...] = field(default_factory=tuple)


S = SectionSpec

DOCS: tuple[DocSpec, ...] = (
    DocSpec("pos-voids-refunds", "Voids, Comps and Refunds", "pos-help", (
        S("Voiding an item before it is sent",
          "Tap the item on the open check and choose Remove. Items that have not been sent to the kitchen "
          "disappear without a trace and need no manager approval.",
          ("How do I remove an item from a check before it goes to the kitchen?",
           "Do I need a manager to delete an unsent item?"),
          ("Remove",)),
        S("Voiding an item after it is sent",
          "Once an item has been fired, a void requires a manager PIN and a reason code such as Wrong Item or "
          "Guest Changed Mind. The void is printed on the kitchen printer as a cancellation ticket and recorded "
          "in the Void Report.",
          ("How can I void a dish that was already sent to the kitchen?",
           "What do I need to cancel an item that has been fired?"),
          ("manager PIN", "reason code")),
        S("Comping items",
          "A comp discounts an item by 100 percent while keeping it in sales reporting as a comp. Servers can "
          "comp up to $15 per check; anything higher needs a manager PIN.",
          ("What is the most a server can comp without a manager?",
           "How do comps work in SousChef POS?"),
          ("$15", "manager PIN")),
        S("Refunding a closed check",
          "Closed checks can be refunded from Check History within 30 days. Card refunds return to the original "
          "card in 3 to 5 business days, while cash refunds open the cash drawer.",
          ("How long do I have to refund a closed check?",
           "How many days does a card refund take to reach the guest?"),
          ("30 days", "3 to 5 business days")),
    )),
    DocSpec("pos-split-checks", "Splitting and Merging Checks", "pos-help", (
        S("Splitting a check by seat",
          "Open the check and tap Split, then choose By Seat. Every item is assigned to the seat it was "
          "ordered for, and each seat becomes its own check that can be paid separately.",
          ("How do I split a bill so each guest pays for their own seat?",
           "Can I divide a check by seat number?"),
          ("By Seat",)),
        S("Splitting a check evenly",
          "Choose Split Evenly and enter the number of guests, up to a maximum of 12. Any leftover cent is added "
          "to the first payment so the total always matches.",
          ("What is the maximum number of guests for an even split?",
           "How do I split a check equally between several people?"),
          ("12", "Split Evenly")),
        S("Merging two checks",
          "Long-press a check and drag it onto another check at the same table to merge them. Checks from "
          "different tables can only be merged by a manager using Transfer Table first.",
          ("How do I combine two checks into one?",
           "Can a server merge checks that belong to different tables?"),
          ("drag", "Transfer Table")),
        S("Moving a check to another table",
          "Tap Transfer Table on the open check and select the new table on the floor plan. Open items, "
          "modifiers and seat assignments move with the check.",
          ("How do I move a party to a different table?",
           "What happens to seats and modifiers when I transfer a table?"),
          ("floor plan",)),
    )),
    DocSpec("pos-printers", "Kitchen Printer and KDS Troubleshooting", "pos-help", (
        S("Kitchen printer is not printing tickets",
          "First check that the printer shows a green status light and that the paper roll is loaded. Then open "
          "Settings, Hardware, Printers and press Test Print; if the test fails, power-cycle the printer and "
          "confirm it is on the same network as the terminal.",
          ("The kitchen printer stopped printing orders, what should I check?",
           "How do I run a test print on a printer?"),
          ("Test Print", "power-cycle")),
        S("Printing to the wrong station",
          "Each menu item is routed to a print station by its Category Routing rule. If drinks print in the "
          "kitchen, open the item's category in Menu Setup and change its Routing to Bar Printer.",
          ("Drinks keep printing in the kitchen instead of at the bar, how do I fix it?",
           "Where do I change which printer an item goes to?"),
          ("Category Routing", "Bar Printer")),
        S("Kitchen display screens freezing or lagging",
          "A kitchen display (KDS) that freezes usually has a weak Wi-Fi signal. Move the access point closer "
          "or connect the display with an Ethernet cable, then restart the KDS app from the screen's menu.",
          ("Why does my KDS screen freeze?",
           "How can I make the kitchen display more reliable?"),
          ("Ethernet", "Wi-Fi")),
        S("Receipt printer prints blank paper",
          "Blank receipts mean the thermal paper is loaded upside down. Open the lid, flip the roll so the "
          "paper unwinds from the bottom, and close the lid until it clicks.",
          ("Why are my receipts coming out blank?",
           "The receipt printer prints white paper, what is wrong?"),
          ("upside down", "flip")),
    )),
    DocSpec("pos-offline-mode", "Offline Mode and Sync", "pos-help", (
        S("What works when the internet is down",
          "Offline Mode keeps taking orders, sending tickets to local printers and accepting cash for up to 72 "
          "hours. Online-only features such as gift card lookups, online orders and cloud reports are paused "
          "until the connection returns.",
          ("Can I keep taking orders if the internet goes out?",
           "How long does offline mode last?"),
          ("72 hours", "cash")),
        S("Taking card payments offline",
          "Card payments taken offline are stored encrypted on the terminal and submitted automatically when "
          "the connection returns. The offline limit is $100 per transaction, and the restaurant carries the "
          "risk if a stored card is later declined.",
          ("Is there a limit on card payments when the system is offline?",
           "Who is responsible if an offline card gets declined later?"),
          ("$100", "stored")),
        S("How syncing works after reconnecting",
          "When the terminal reconnects it uploads queued orders in the order they were created and then "
          "downloads menu changes. A Sync Complete banner appears at the top of the screen; large backlogs can "
          "take up to 10 minutes.",
          ("What happens when my terminal comes back online?",
           "How do I know the offline orders finished syncing?"),
          ("Sync Complete", "10 minutes")),
    )),
    DocSpec("pos-tips-payroll", "Tips, Tip Pooling and Payroll Export", "pos-help", (
        S("Adding tips after payment",
          "Servers can adjust a tip on a card payment until the batch closes, which happens automatically at "
          "4:00 AM. After the batch closes the tip can no longer be changed from the POS.",
          ("Until when can I change a tip on a card payment?",
           "At what time does the batch close and lock tips?"),
          ("4:00 AM",)),
        S("Setting up a tip pool",
          "In Settings, Tips, create a pool and assign roles with a point weight. For example, servers at 10 "
          "points, bussers at 4 points and bartenders at 6 points share the pooled tips in proportion to "
          "points multiplied by hours worked.",
          ("How do I configure tip pooling for servers and bussers?",
           "How are pooled tips divided between roles?"),
          ("point weight", "hours worked")),
        S("Exporting payroll hours and tips",
          "Open Reports, Payroll and choose a pay period, then press Export. SousChef supports CSV and direct "
          "exports to Gusto and ADP; hours and tips are listed per employee.",
          ("Can I export staff hours and tips to my payroll provider?",
           "Which payroll systems does SousChef export to?"),
          ("CSV", "Gusto", "ADP")),
        S("Tip-out for support staff",
          "A Tip-Out Rule takes a percentage of a server's tips and credits it to a role. The default is 3 "
          "percent of net sales to the bar and 2 percent to the host, and rules can be edited by owners only.",
          ("What is the default tip-out percentage for the bar?",
           "Who is allowed to edit tip-out rules?"),
          ("3 percent", "owners")),
    )),
    DocSpec("pos-inventory", "Inventory and Par Levels", "pos-help", (
        S("Setting par levels",
          "A par level is the quantity you want on the shelf after a delivery. Enter it on the ingredient "
          "page, and SousChef flags the ingredient on the Low Stock list when counts drop below 25 percent "
          "of par.",
          ("What is a par level?",
           "When does an ingredient show up on the low stock list?"),
          ("25 percent",)),
        S("Recipe costing",
          "Link ingredients to a menu item through its recipe card. Food cost percentage is calculated as "
          "ingredient cost divided by menu price and updates whenever you enter a new supplier invoice price.",
          ("How is food cost percentage calculated?",
           "How do I link ingredients to a dish to see its cost?"),
          ("recipe card", "divided by menu price")),
        S("Counting stock",
          "Start a Stock Count from Inventory, Counts and walk through the storage areas in order. Variances "
          "above 5 percent are highlighted in red so you can investigate waste or theft.",
          ("How do I do a physical inventory count?",
           "What does a red variance mean in a stock count?"),
          ("Stock Count", "5 percent")),
        S("Ordering from suppliers",
          "Press Create Purchase Order on the Low Stock list to build an order from your par levels. Orders "
          "can be emailed or sent as a PDF, and received quantities update stock when you mark the delivery as "
          "received.",
          ("Can SousChef create purchase orders for my suppliers?",
           "How do received deliveries update my stock?"),
          ("Purchase Order", "received")),
    )),
    DocSpec("pos-menu-editing", "Editing Menu Items, Modifiers and 86ing", "pos-help", (
        S("Adding a new menu item",
          "Open Menu Setup, choose a category and press Add Item. Enter a name, price and tax group; new "
          "items appear on every terminal within 60 seconds.",
          ("How do I add a new dish to the POS menu?",
           "How quickly do menu changes show up on the terminals?"),
          ("Add Item", "60 seconds")),
        S("Creating modifiers",
          "Modifiers such as Extra Cheese or No Onion are grouped in a modifier group. Set a group to Required "
          "when the server must pick an option, and add an upcharge to any option that costs extra.",
          ("How do I make a required modifier like a steak temperature?",
           "How can I charge extra for a modifier?"),
          ("Required", "upcharge")),
        S("86ing an item",
          "When the kitchen runs out of something, long-press the item and choose 86. The item is greyed out on "
          "all terminals and online ordering immediately, and it returns automatically at the next day's "
          "opening unless you un-86 it earlier.",
          ("What does it mean to 86 an item and how do I do it?",
           "When does an 86'd item come back on the menu?"),
          ("long-press", "next day")),
        S("Scheduling happy hour pricing",
          "Create a Price Schedule under Menu Setup and select the days, start time and end time. Items "
          "listed in the schedule switch to their happy hour price automatically and switch back at the end "
          "time.",
          ("Can I set up a happy hour price that switches on automatically?",
           "Where do I schedule time-based pricing?"),
          ("Price Schedule",)),
    )),
    DocSpec("pos-user-permissions", "Staff Roles, PINs and Permissions", "pos-help", (
        S("Adding an employee",
          "Go to Staff, press Add Employee, and enter a name, role and four-digit PIN. The employee can clock "
          "in on any terminal right after saving.",
          ("How do I add a new server to the system?",
           "How many digits is an employee PIN?"),
          ("four-digit PIN",)),
        S("Default roles and permissions",
          "SousChef ships with four roles: Server, Bartender, Manager and Owner. Only Managers and Owners can "
          "approve voids after firing, open the cash drawer without a sale, and change prices.",
          ("Which roles are available by default?",
           "Who can open the cash drawer without making a sale?"),
          ("Server", "Manager", "Owner")),
        S("Resetting a forgotten PIN",
          "A manager can reset any employee PIN from Staff by pressing Reset PIN, which sets a temporary code "
          "that must be changed at the next login. Managers cannot reset the Owner PIN; the owner must use the "
          "email recovery link.",
          ("An employee forgot their PIN, how do I reset it?",
           "How does the owner recover their own PIN?"),
          ("Reset PIN", "email recovery link")),
        S("Locking out former employees",
          "Set an employee to Inactive in Staff to block logins immediately while keeping their sales history "
          "and tip records. Inactive employees can be reactivated at any time.",
          ("How do I stop a former employee from logging in without deleting their data?",
           "Can I reactivate an inactive employee?"),
          ("Inactive",)),
    )),
    DocSpec("pos-payments", "Card Payments, Fees and Chargebacks", "pos-help", (
        S("Processing fees",
          "SousChef Payments charges 2.6 percent plus 10 cents per in-person card transaction and 2.9 percent "
          "plus 30 cents for keyed-in or online payments. There are no monthly minimums or PCI fees.",
          ("What are the card processing rates?",
           "Is there a monthly minimum or PCI fee for payments?"),
          ("2.6 percent", "10 cents")),
        S("Payout schedule",
          "Card sales are deposited to your bank account the next business day when the batch closes before "
          "4:00 AM. Weekend sales arrive on Monday, and Instant Payout is available for a 1 percent fee.",
          ("When do I get my card sales deposited in my bank account?",
           "How much does instant payout cost?"),
          ("next business day", "1 percent")),
        S("Responding to a chargeback",
          "When a guest disputes a charge you receive an email and a Disputes notice in the dashboard. Upload "
          "the signed receipt and the itemized check within 7 days, otherwise the dispute is lost "
          "automatically.",
          ("A guest disputed a charge, what do I do?",
           "How long do I have to answer a chargeback?"),
          ("7 days", "signed receipt")),
        S("Tap to pay and chip readers",
          "The Countertop Reader accepts chip, tap and mobile wallets. If the reader says Card Read Error, "
          "ask the guest to insert the chip instead of tapping, or clean the contacts with a dry cloth.",
          ("The card reader says Card Read Error, how can I fix it?",
           "Does the reader accept Apple Pay and tap cards?"),
          ("insert the chip", "mobile wallets")),
    )),
    DocSpec("pos-reports", "Reports and End-of-Day Close", "pos-help", (
        S("Running the end-of-day close",
          "Press End of Day in the Manager menu after all checks are closed. The close prints the Z report, "
          "locks the day's sales and automatically closes the card batch.",
          ("How do I close out the day?",
           "What does the end of day close do?"),
          ("End of Day", "Z report")),
        S("Sales by category report",
          "Reports, Sales, Category Mix shows revenue and item counts per menu category. Use the date range "
          "picker to compare weeks, and press Export to download the data as a CSV file.",
          ("Where can I see how much each menu category sold?",
           "Can I download sales reports as a spreadsheet?"),
          ("Category Mix", "CSV")),
        S("Finding why the till is off",
          "Open Reports, Cash Drawer to compare expected cash with the counted amount. The report lists every "
          "no-sale opening, paid-in and paid-out, so you can trace a shortage back to a single transaction.",
          ("My cash drawer count doesn't match, where do I investigate?",
           "How do I see every time the drawer was opened without a sale?"),
          ("Cash Drawer", "no-sale")),
        S("Scheduled report emails",
          "Under Reports, Schedules you can email the daily sales summary to up to 5 recipients at a time you "
          "choose. Owners receive it at 6:00 AM by default.",
          ("Can I get the daily sales summary by email automatically?",
           "How many people can receive a scheduled report?"),
          ("5 recipients", "6:00 AM")),
    )),
    DocSpec("menu-bistro", "Maison Basil Bistro Dinner Menu", "menu", (
        S("Starters",
          "- French Onion Soup ($9): caramelized onions, beef broth and melted Gruyere. Contains: dairy, gluten.\n"
          "- Burrata Salad ($14): burrata, heirloom tomatoes and basil oil. Contains: dairy.\n"
          "- Crispy Calamari ($13): lightly fried squid with lemon aioli. Contains: gluten, egg, shellfish.",
          ("How much is the French onion soup?",
           "Which starter contains shellfish?"),
          ("$9", "Crispy Calamari")),
        S("Mains",
          "- Steak Frites ($32): 10 oz hanger steak, shallot butter and fries. Contains: dairy.\n"
          "- Mushroom Risotto ($24): arborio rice, wild mushrooms and parmesan. Vegetarian. Contains: dairy.\n"
          "- Pan-Seared Salmon ($29): crispy skin salmon over lentils with dill cream. Contains: fish, dairy.",
          ("What is the price of the steak frites?",
           "Is there a vegetarian main course at Maison Basil?"),
          ("$32", "Mushroom Risotto")),
        S("Desserts",
          "- Crème Brûlée ($10): vanilla custard with a caramelized sugar crust. Contains: dairy, egg. "
          "Gluten free.\n"
          "- Chocolate Fondant ($11): warm chocolate cake with a molten centre. Contains: dairy, egg, gluten.\n"
          "- Lemon Sorbet ($7): house-made sorbet. Vegan and gluten free.",
          ("Which desserts are gluten free?",
           "Do you have a vegan dessert?"),
          ("Crème Brûlée", "Lemon Sorbet")),
        S("Dietary and allergen policy",
          "The kitchen handles wheat, dairy, nuts, egg, fish and shellfish, so cross-contact is possible. "
          "Gluten-free pasta can be substituted in any pasta dish for a $3 upcharge. Please tell your server "
          "about allergies before ordering.",
          ("Can I swap in gluten-free pasta and what does it cost?",
           "Is there a risk of cross-contact with allergens in the kitchen?"),
          ("$3", "cross-contact")),
    )),
    DocSpec("menu-brunch", "Maison Basil Weekend Brunch Menu", "menu", (
        S("Brunch hours and bottomless drinks",
          "Brunch is served Saturday and Sunday from 10:00 AM to 2:30 PM. Bottomless mimosas are $22 per "
          "person with a 90 minute limit and require a food order.",
          ("What time is weekend brunch served?",
           "How much are the bottomless mimosas and how long can I drink them?"),
          ("10:00 AM", "$22", "90 minute")),
        S("Brunch plates",
          "- Eggs Benedict ($17): poached eggs, Canadian bacon and hollandaise on an English muffin. Contains: "
          "egg, dairy, gluten.\n"
          "- Buttermilk Pancakes ($14): three pancakes with maple butter and berries. Contains: dairy, egg, "
          "gluten.\n"
          "- Shakshuka ($16): eggs baked in spiced tomato sauce with feta and grilled bread. Vegetarian.",
          ("How much is the eggs benedict?",
           "Which brunch dish is baked in a tomato sauce?"),
          ("$17", "Shakshuka")),
        S("Kids brunch",
          "Children under 10 can order the Kids Brunch for $9, which includes one pancake or scrambled eggs, "
          "fruit and a small juice. Refills of juice are free.",
          ("Is there a children's brunch option and how much is it?",
           "What does the kids brunch come with?"),
          ("$9", "small juice")),
    )),
    DocSpec("faq-billing", "Billing and Subscription FAQ", "faq", (
        S("Subscription plans and prices",
          "SousChef POS offers three plans: Starter at $69 per month, Pro at $149 per month and Enterprise "
          "with custom pricing for groups of 5 or more locations. Each plan includes unlimited menu items and "
          "staff accounts.",
          ("How much does the Pro plan cost per month?",
           "What is the price of the Starter plan?"),
          ("$149", "$69")),
        S("Free trial",
          "Every new account gets a 14 day free trial of the Pro plan without entering a credit card. At the "
          "end of the trial the account drops to read-only mode until you pick a plan.",
          ("Is there a free trial and do I need a credit card?",
           "What happens when my trial ends?"),
          ("14 day", "read-only")),
        S("Cancelling a subscription",
          "Owners can cancel any time under Account, Billing, Cancel Plan. Service continues until the end of "
          "the paid month, and exported data stays downloadable for 90 days.",
          ("How do I cancel my SousChef subscription?",
           "How long can I download my data after cancelling?"),
          ("Cancel Plan", "90 days")),
        S("Hardware purchases and warranty",
          "Terminals, printers and card readers carry a 12 month warranty. Defective hardware is replaced "
          "with a pre-paid return label, and shipping takes 2 to 4 business days.",
          ("What warranty comes with the POS terminals?",
           "How long does a replacement terminal take to ship?"),
          ("12 month", "2 to 4 business days")),
    )),
    DocSpec("faq-online-orders", "Online Ordering and Delivery FAQ", "faq", (
        S("Setting up online ordering",
          "Turn on Online Ordering under Channels and SousChef creates a free ordering page for your "
          "restaurant. Online menu items sync from the POS, and online orders print on the kitchen printer "
          "automatically.",
          ("How do I start taking online orders?",
           "Do online orders print in my kitchen?"),
          ("Channels", "kitchen printer")),
        S("Online ordering fees",
          "There is no commission on orders placed through your own ordering page; only the standard card "
          "processing rate applies. Orders accepted through third-party marketplaces are charged the "
          "marketplace's own commission.",
          ("Does SousChef charge commission on online orders?",
           "Do marketplace orders have extra fees?"),
          ("no commission", "marketplace")),
        S("Pickup and prep times",
          "Set a default prep time between 10 and 60 minutes under Channels, Prep Time. Guests see the "
          "estimated pickup time at checkout, and you can add 15 minutes with one tap when the kitchen is "
          "busy.",
          ("How do I change the estimated pickup time for online orders?",
           "Can I slow down online orders when the kitchen is slammed?"),
          ("10 and 60 minutes", "15 minutes")),
        S("Delivery radius",
          "Restaurants can deliver with their own drivers within a radius of up to 8 miles. Set the radius "
          "and the delivery fee in Channels, Delivery, and orders outside the radius are rejected at "
          "checkout.",
          ("How far can I deliver with my own drivers?",
           "Where do I set the delivery fee?"),
          ("8 miles", "Channels, Delivery")),
    )),
)

# Questions the knowledge base cannot answer; the assistant must refuse.
OUT_OF_SCOPE_QUESTIONS: tuple[str, ...] = (
    "What is the capital of France?",
    "Who won the football world cup in 2018?",
    "Can you write me a poem about the ocean?",
    "How do I reset my Netflix password?",
    "What is the weather forecast for tomorrow in Chicago?",
    "Should I invest in Bitcoin this year?",
    "How do I change the oil in my car?",
    "Translate good morning into Japanese.",
    "Who is the president of Brazil?",
    "Recommend a good laptop for video editing.",
    "How many calories are in a Big Mac?",
    "What is the square root of 1764?",
    "Which airline has the cheapest flights to Tokyo?",
    "Explain how photosynthesis works.",
)
