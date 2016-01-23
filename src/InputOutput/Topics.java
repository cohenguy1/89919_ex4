/* Ido Cohen	Guy Cohen	203516992	304840283 */
package InputOutput;

public enum Topics {
	ACQ("acq"),
	MONEY_FX("money-fx"),
	GRAIN("grain"),
	CRUDE("crude"),
	TRADE("trade"),
	INTEREST("interest"),
	SHIP("ship"),
	WHEAT("wheat"),
	CORN("corn");

	private String text;

	Topics(String text) {
		this.text = text;
	}

	public String getText() {
		return this.text;
	}
	
	public static Topics fromNumber(int index) {
		switch (index) 
		{
		case 0:
			return Topics.ACQ;
		case 1:
			return Topics.MONEY_FX;
		case 2:
			return Topics.GRAIN;
		case 3:
			return Topics.CRUDE;
		case 4:
			return Topics.TRADE;
		case 5:
			return Topics.INTEREST;
		case 6:
			return Topics.SHIP;
		case 7:
			return Topics.WHEAT;
		case 8:
			return Topics.CORN;
		default:
			return Topics.ACQ;	
		}
	}
	
	public static int getNumberOfTopcis() {
		return 9; //Change if topic is added!
	}

	public static Topics fromString(String text) {
		if (text != null) {
			for (Topics b : Topics.values()) {
				if (text.equalsIgnoreCase(b.text)) {
					return b;
				}
			}
		}
		return null;
	}


}
