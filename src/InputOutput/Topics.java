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
